import argparse
import os
import random

import monai.transforms as monai_transforms
import torch
import torchvision.transforms.v2 as v2


ALL = {
    "identity": v2.Lambda(lambda x: x),
    "random_crop": v2.RandomCrop(224),
    "horizontal_flip": v2.RandomHorizontalFlip(p=1.0),
    "vertical_flip": v2.RandomVerticalFlip(p=1.0),
    "rotation": v2.RandomRotation(degrees=30),
    "translate_x": v2.RandomAffine(degrees=0, translate=[0.2, 0]),
    "translate_y": v2.RandomAffine(degrees=0, translate=[0, 0.2]),
    "shear_x": v2.RandomAffine(degrees=0, shear=[0.0, 30.0]),
    "shear_y": v2.RandomAffine(degrees=0, shear=[0.0, 0.0, 0.0, 30.0]),
    "brightness": v2.ColorJitter(brightness=0.5),
    "contrast": v2.ColorJitter(contrast=0.5),
    "saturation": v2.ColorJitter(saturation=0.5),
    "gaussian_blur": v2.GaussianBlur(kernel_size=3),
    "equalize": v2.RandomEqualize(p=1.0),
    "median_blur": monai_transforms.MedianSmooth(radius=3),
    "grid_distortion": monai_transforms.RandGridDistortion(prob=1.0),
    "gaussian_noise": monai_transforms.RandGaussianNoise(prob=1.0),
    "scaling": v2.RandomAffine(
        degrees=0, scale=[0.8, 1.2]
    ),  # Sensible range based on prior works
    "elastic_transform": v2.ElasticTransform(),
}

PHOTOMETRIC = {
    "identity": v2.Lambda(lambda x: x),
    "brightness": v2.ColorJitter(brightness=0.5),
    "contrast": v2.ColorJitter(contrast=0.5),
    "saturation": v2.ColorJitter(saturation=0.5),
    "gaussian_blur": v2.GaussianBlur(kernel_size=3),
    "equalize": v2.RandomEqualize(p=1.0),
    "median_blur": monai_transforms.MedianSmooth(radius=3),
    "gaussian_noise": monai_transforms.RandGaussianNoise(prob=1.0),
}

GEOMETRIC = {
    "identity": v2.Lambda(lambda x: x),
    "random_crop": v2.RandomCrop(224),
    "horizontal_flip": v2.RandomHorizontalFlip(p=1.0),
    "vertical_flip": v2.RandomVerticalFlip(p=1.0),
    "rotation": v2.RandomRotation(degrees=30),
    "translate_x": v2.RandomAffine(degrees=0, translate=[0.2, 0]),
    "translate_y": v2.RandomAffine(degrees=0, translate=[0, 0.2]),
    "shear_x": v2.RandomAffine(degrees=0, shear=[0.0, 30.0]),
    "shear_y": v2.RandomAffine(degrees=0, shear=[0.0, 0.0, 0.0, 30.0]),
    "grid_distortion": monai_transforms.RandGridDistortion(prob=1.0),
    "scaling": v2.RandomAffine(
        degrees=0, scale=[0.8, 1.2]
    ),  # Sensible range based on prior works
    "elastic_transform": v2.ElasticTransform(),
}

class TrivialAugment(torch.nn.Module):
    def __init__(self, num_ops, transforms) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.transforms = transforms

    def forward(self, x):
        # Randomly sample N operations without replacement
        ops = random.sample(list(self.transforms.values()), self.num_ops)

        # Apply each operation in sequence
        for op in ops:
            x = op(x)

        return x

def get_augmentation_set(subset):
    if subset == "all":
        return ALL
    elif subset == "photometric":
        return PHOTOMETRIC
    elif subset == "geometric":
        return GEOMETRIC


def objective(trial, args):
    """Evaluate a set of hyperparameter values using cross-validation."""
    num_ops = trial.suggest_categorical("num_ops", [1, 2, 3, 4, 5])

    # Define the train and val transforms
    augmentation_set = get_augmentation_set(args.dataset, args.subset)
    train_transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(224, antialias=True),
            TrivialAugment(num_ops, augmentation_set),
            v2.CenterCrop(
                224
            ),  # Only makes a difference if random crop wasn't performed
            v2.Normalize(args.mean, args.std),
        ]
    )
    val_transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(224, antialias=True),
            v2.CenterCrop(224),
            v2.Normalize(args.mean, args.std),
        ]
    )

    # 5 x 2 cross validation
    balanced_accuracies = []
    for split, train_fold, train_subset, val_subset in get_dataset_folds(
        args, train_transform, val_transform
    ):
        pl.seed_everything(args.seed)

        # Define data loaders for training and validation data
        train_loader = DataLoader(
            train_subset,
            batch_size=64,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=args.workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=args.workers,
        )

        # Calculate the class weights
        counts = np.bincount(
            (
                train_subset.dataset.targets
                if type(train_subset) == Subset
                else train_subset.targets
            ),
            minlength=args.num_classes,
        )
        class_weights = get_class_balanced_weights(counts)

        # Instantiate the model
        lit_model = LitImageClassifier(
            resnet18, num_classes=args.num_classes, class_weights=class_weights
        )

        # Setup the trainer
        checkpoint_callback = ModelCheckpoint(
            monitor="val/bacc_epoch", mode="max", save_top_k=1
        )
        trainer = pl.Trainer(
            max_epochs=100,
            deterministic=False,
            default_root_dir=os.path.join(
                args.log_dir,
                f"ops_{num_ops}",
                f"split_{split}",
                f"trainfold_{train_fold}",
            ),
            fast_dev_run=args.fast_dev_run,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback],
        )

        # Train the model
        trainer.fit(lit_model, train_loader, val_loader)
        balanced_accuracies.append(lit_model.val_bacc_best.item())

    # Return a simple mean and std. err. of the balanced accuracies just for Optuna
    return np.mean(balanced_accuracies), stats.sem(balanced_accuracies)


def main():
    args = parse_args()

    # Configure Optuna storage
    os.makedirs(args.log_dir, exist_ok=True)
    storage_path = os.path.join(args.log_dir, f"{args.dataset}.log")
    storage = JournalStorage(JournalFileStorage(storage_path))

    # Configure study
    study = optuna.create_study(
        sampler=optuna.samplers.BruteForceSampler(),
        study_name=args.dataset,
        storage=storage,
        directions=["maximize", "minimize"],
        load_if_exists=True,
    )

    # Run the study
    study.optimize(
        lambda trial: objective(trial, args), n_trials=1
    )  # Use n_trials=5 for full study

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    for trial in study.best_trials:
        print("  Value: {}".format(trial.values))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
