import torch
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torchvision.transforms.v2 as v2
import pandas as pd

import time
import platform
from IPython.display import display, HTML


def train_model(model, loss_fn, epochs = 5, use_mixup = False, optimizer=None, scheduler=None, train_loader=None,
                val_loader=None, metrics=None, data_module=None,
                save_model_filename=None, load_model_filename=None,
                update_pct_interval=5, max_epochs_display=5, resume_training=False,
                pause_before_train = 5
               ):
    """
    Trains a PyTorch model, with options to load a pre-trained model and
    resume training from a checkpoint.

    Parameters:
    - model: The PyTorch model to be trained.
    - loss_fn: Loss function.
    - optimizer: Optimizer for training.
    - epochs: Number of epochs to train.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - metrics: Metrics to calculate during training.
    - data_module: Data module for handling data loading.
    - save_model_filename: Path to save the best model.
    - load_model_filename: Path to a pre-trained model checkpoint.
    - update_pct_interval: Update display percentage interval.
    - max_epochs_display: How many last epochs to display metrics for.
    - resume_training: Whether to resume training from the checkpoint.
    - pause_before_train: number of seconds to wait to view initial messages.  Default 5.
    """
    
    # Initialize the wrapped model
    if not use_mixup:
         pl_model = LitBasicModel(model, loss_fn, optimizer, metrics, scheduler)
    else:
        pl_model = ClassifierMixupModel(model, loss_fn, optimizer, metrics, scheduler, use_mixup = use_mixup)
    
    # Load model state from checkpoint if not resuming entire training
    if load_model_filename and not resume_training:
        checkpoint = torch.load(load_model_filename, map_location=lambda storage, loc: storage)
        model_state_dict = checkpoint['state_dict']
        # Adjust for the 'model.' prefix used by PyTorch Lightning
        adjusted_model_state_dict = {k.replace('model.', ''): v for k, v in model_state_dict.items()}
        model.load_state_dict(adjusted_model_state_dict)
    
    # Setup callbacks
    print_progress_cb = PrintProgressMetricsCallback(update_percent=update_pct_interval,
                                                     max_epochs_display=max_epochs_display,
                                                     pause_before_train = pause_before_train)
    callbacks = [print_progress_cb]
    
    # Conditionally add a ModelCheckpoint callback
    if save_model_filename:
        monitor = "val_loss" if val_loader is not None else "train_loss"
        checkpoint_callback = ModelCheckpoint(monitor=monitor, dirpath=".",
                                              filename=save_model_filename, save_top_k=1, mode="min")
        callbacks.append(checkpoint_callback)

    # Initialize the Trainer
    trainer = Trainer(max_epochs=epochs, 
                      callbacks=callbacks, 
                      enable_progress_bar=False,
                      enable_checkpointing=True if save_model_filename else False,
                      num_sanity_val_steps=0,
                     )

    # Determine the checkpoint path for resuming training
    ckpt_path = None
    if resume_training and load_model_filename:
        ckpt_path = load_model_filename

    # Fit the model
    if data_module:
        trainer.fit(pl_model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

    # Optionally, return metrics and the trained model
    metrics_df = print_progress_cb.metrics_df

    model = pl_model.model

    return metrics_df

####################

class LitBasicModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer=None, metrics=None, scheduler=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        
        # If optimizer is None, initialize AdamW with model parameters and a default lr of 0.001
        self.optimizer = AdamW(self.model.parameters(), lr=0.001) if optimizer is None else optimizer
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        
        # Dynamically add metrics as modules
        for metric_name, metric in self.metrics.items():
            self.add_module(metric_name, metric)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1] # some of our dataloaders return extra
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        for name, metric in self.metrics.items():
            self.log(f'train_{name}', metric(logits, y).item(), on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=False)
        if self.metrics:
            for name, metric in self.metrics.items():
                self.log(f'val_{name}', metric(logits, y).item(), on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        if self.scheduler:
            return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler, 'interval': 'step'}}
        else:
            return self.optimizer

class LitClassifierMixupModel(LitBasicModel):
    def __init__(self, model, loss_fn, optimizer=None, metrics=None, scheduler=None, use_mixup=False):
        super().__init__(model, loss_fn, optimizer, metrics, scheduler)

        self.use_mixup = use_mixup
    
        # find number of outputs
        last_linear_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear_layer = module
        num_classes = last_linear_layer.out_features

        # configure mixup if using
        if self.use_mixup:
            cutmix = v2.CutMix(num_classes=num_classes)
            mixup = v2.MixUp(num_classes=num_classes)
            self.mixup_func = v2.RandomChoice([cutmix, mixup])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1] # some of our dataloaders return extra
        if self.use_mixup:
            x, y = self.mixup_func(x,y)
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        if not self.use_mixup and self.metrics:
            for name, metric in self.metrics.items():
                self.log(f'train_{name}', metric(logits, y).item(), on_step=False, on_epoch=True, prog_bar=False)
        return loss

class ClassifierMixupModel(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer=None, metrics=None, scheduler=None, use_mixup=False):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        
        # If optimizer is None, initialize AdamW with model parameters and a default lr of 0.001
        self.optimizer = AdamW(self.model.parameters(), lr=0.001) if optimizer is None else optimizer
        self.scheduler = scheduler
        self.metrics = metrics if metrics is not None else {}
        
        # Dynamically add metrics as modules
        for metric_name, metric in self.metrics.items():
            self.add_module(metric_name, metric)
        self.use_mixup = use_mixup

        # find number of outputs
        last_linear_layer = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                last_linear_layer = module
        num_classes = last_linear_layer.out_features

        # configure mixup if using
        if self.use_mixup:
            cutmix = v2.CutMix(num_classes=num_classes)
            mixup = v2.MixUp(num_classes=num_classes)
            self.mixup_func = v2.RandomChoice([cutmix, mixup])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1] # some of our dataloaders return extra
        if self.use_mixup:
            x, y = self.mixup_func(x,y)
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=False)
        if not self.use_mixup and self.metrics:
            for name, metric in self.metrics.items():
                self.log(f'train_{name}', metric(logits, y).item(), on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss.item(), on_epoch=True, prog_bar=False)
        if self.metrics:
            for name, metric in self.metrics.items():
                self.log(f'val_{name}', metric(logits, y).item(), on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        if self.scheduler:
            return {'optimizer': self.optimizer, 'lr_scheduler': {'scheduler': self.scheduler, 'interval': 'step'}}
        else:
            return self.optimizer

class PrintProgressMetricsCallback(Callback):

    def __init__(self, update_percent = 10, max_epochs_display = 10, pause_before_train=5):
        super().__init__()
        self.metrics_df = pd.DataFrame()
        self.training_step_total = 0
        self.validation_step_total = 0
        self.validation_elapsed_time = 0 #in case there is no validation step
        self.validation_percent_complete = 0
        self.update_percent = update_percent
        self.max_epochs_display = max_epochs_display
        self.pause_before_train = pause_before_train

    def on_train_start(self, trainer, pl_module):
        # Add a pause at the start of training (default 5 seconds)
        print(f'\n Training starts in {self.pause_before_train} seconds ...')
        time.sleep(self.pause_before_train)

    def _update_display(self, trainer, pl_module):
        clear_output() #this is our own function to work in multiple evirons
        epoch_str = f'Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}'
        training_percent_str = f'Training {self.training_percent_complete:0.2f}% complete'
        display_str = f'{epoch_str}, {training_percent_str}'

        # Check if validation step total is greater than 0 to determine if validation is being performed
        if self.validation_step_total > 0:
            validation_percent_str = f'Validation {self.validation_percent_complete:0.2f}% complete'
            display_str += f', {validation_percent_str}'

        lr = trainer.optimizers[0].param_groups[0]['lr']
        display_str += f' lr = {lr:0.3e}'
        
        print(display_str)
        if not self.metrics_df.empty:
            display_dataframe(self.metrics_df.tail(self.max_epochs_display))

    def on_train_epoch_start(self, trainer, pl_module):
        self.training_start_time = time.time()
        self.training_elapsed_time = 0
        self.training_step_total = len(trainer.train_dataloader)
        self.training_step_counter = 0
        self.training_percent_complete = 0
        if self.validation_step_total > 0:
            self.validation_percent_complete = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.validation_start_time = time.time()
        self.validation_elapsed_time = 0
        if trainer.num_val_batches:
            self.validation_step_total = sum(trainer.num_val_batches)
        else:
            self.validation_step_total = 0
        self.validation_step_counter = 0
        self.validation_percent_complete = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.training_step_counter += 1
        self.training_percent_complete = (self.training_step_counter / self.training_step_total) * 100
        if self.training_percent_complete % self.update_percent <= (1 / self.training_step_total) * 100:
            self._update_display(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.validation_step_counter += 1
        self.validation_percent_complete = (self.validation_step_counter / self.validation_step_total) * 100
        if self.validation_percent_complete % self.update_percent <= (1 / self.validation_step_total) * 100:
            self._update_display(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_elapsed_time += time.time() - self.validation_start_time

    def on_train_epoch_end(self, trainer, pl_module):
        new_row = pd.DataFrame({'Epoch':[trainer.current_epoch + 1]})
        self.epoch_metrics = {key: [value.item()] for key, value in sorted(trainer.logged_metrics.items())}
        for col,data in self.epoch_metrics.items():
            new_row[col] = data
        self.training_elapsed_time += time.time() - self.training_start_time
        new_row['Time'] = [self.training_elapsed_time+self.validation_elapsed_time]
        lr = trainer.optimizers[0].param_groups[0]['lr']
        new_row['LR'] =[lr]
        self.metrics_df = pd.concat([self.metrics_df, new_row],ignore_index=True)
        self._update_display(trainer, pl_module)

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        return False

# Function to clear the output in the notebook or console
def clear_output():
    if in_notebook():
        from IPython.display import clear_output as clear
        clear(wait=True)
    else:
        os_name = platform.system()
        if os_name == 'Windows':
            os.system('cls')
        else:
            os.system('clear')

def display_dataframe(df):
    print(df.to_string(index=False))