from fastai.vision.all import *
import pandas as pd
from pathlib import Path


# BUSIS dataset


def busis_img_depad(img_name):
    img_name = str(img_name)
    img_index = img_name[4:]
    img_index = int(img_index)
    return "case" + str(img_index)


def busis_get_class(row):
    image_stem = row.stem
    tumor_type = busis_classes[busis_classes["img name"] == image_stem][
        "tumor type"
    ].values[0]
    if tumor_type == "B":
        return "benign"
    elif tumor_type == "M":
        return "malignant"
    else:
        return None


def busis_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    label_name = busis_dataset / "GT" / f"{fn.stem}_GT.bmp"
    return [label_name]


def busis_dataset_make(dv):
    global busis_dataset
    busis_dataset = Path(dv + "/BUS Project Home/Datasets/BUSIS")

    global busis_classes
    busis_classes = pd.read_csv(busis_dataset / "BUSIS562.csv")[
        ["img name", "Ground Truth Name", "tumor type"]
    ]
    busis_classes.dropna(inplace=True)
    busis_classes["img name"] = busis_classes["img name"].apply(busis_img_depad)

    busis_all_images = [f for f in (busis_dataset / "Original").glob("*")]
    busis_all_masks = [busis_get_label(f) for f in busis_all_images]
    busis_df = pd.DataFrame(
        {
            "dataset": ["BUSIS"] * len(busis_all_images),
            "image": busis_all_images,
            "mask": busis_all_masks,
        }
    )
    busis_df["tumor"] = busis_df["image"].apply(busis_get_class)
    return busis_df


# BUS Dataset B


def get_bus_dataset_b_class(row):
    image_stem = int(row.stem)
    tumor_type = bus_dataset_b_classes[bus_dataset_b_classes["Image"] == image_stem][
        "Type"
    ].values[0]
    if tumor_type == "Benign":
        return "benign"
    elif tumor_type == "Malignant":
        return "malignant"
    else:
        return None
        

def bus_dataset_b_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    label_name = bus_dataset_b / "GT" / f"{fn.stem:0>6}.png"
    return [label_name]


def bus_dataset_b_make(dv):
    global bus_dataset_b
    bus_dataset_b = Path(dv + "/BUS Project Home/Datasets/BUS_Dataset_B")

    bus_dataset_b_all_images = [f for f in (bus_dataset_b / "original").glob("*")]
    bus_dataset_b_all_masks = [
        bus_dataset_b_get_label(f) for f in bus_dataset_b_all_images
    ]
    bus_dataset_b_df = pd.DataFrame(
        {
            "dataset": ["BUS_Dataset_B"] * len(bus_dataset_b_all_images),
            "image": bus_dataset_b_all_images,
            "mask": bus_dataset_b_all_masks,
        }
    )

    global bus_dataset_b_classes
    bus_dataset_b_classes = pd.read_excel(bus_dataset_b / "DatasetB.xlsx")
    bus_dataset_b_df["tumor"] = bus_dataset_b_df["image"].apply(get_bus_dataset_b_class)
    return bus_dataset_b_df


# Dataset BUSI with GT


def dataset_busi_with_gt_get_label(fn, image_class):
    """
    Given an image file name and a folder path,
    returns the paths to all corresponding masks.
    """
    mask_folder = image_class
    return [
        f
        for f in (dataset_busi_with_gt / mask_folder).glob("*")
        if fn.stem in str(f) and "_mask" in str(f)
    ]


def dataset_busi_with_gt_make(dv):
    global dataset_busi_with_gt
    dataset_busi_with_gt = Path(dv + "/BUS Project Home/Datasets/Dataset_BUSI_with_GT")
    dataset_busi_with_gt_all_images = []
    dataset_busi_with_gt_all_masks = []
    tumor_labels = []
    for image_class in ["benign", "malignant", "normal"]:
        dataset_busi_with_gt_images = [
            f
            for f in (dataset_busi_with_gt / image_class).glob("*")
            if "_mask" not in str(f)
        ]
        dataset_busi_with_gt_masks = [
            dataset_busi_with_gt_get_label(f, image_class)
            for f in dataset_busi_with_gt_images
        ]
        dataset_busi_with_gt_all_images = (
            dataset_busi_with_gt_all_images + dataset_busi_with_gt_images
        )
        dataset_busi_with_gt_all_masks = (
            dataset_busi_with_gt_all_masks + dataset_busi_with_gt_masks
        )
        tumor_labels = tumor_labels + [image_class] * len(dataset_busi_with_gt_images)
    dataset_busi_with_gt_df = pd.DataFrame(
        {
            "dataset": ["Dataset_BUSI_with_GT"] * len(dataset_busi_with_gt_all_images),
            "image": dataset_busi_with_gt_all_images,
            "mask": dataset_busi_with_gt_all_masks,
            "tumor": tumor_labels,
        }
    )
    
    dataset_busi_with_gt_df = dataset_busi_with_gt_df[dataset_busi_with_gt_df['tumor'] != 'normal']
    
    return dataset_busi_with_gt_df


# Mayo dataset


def mayo_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    mask_folder = mayo_dataset / "Cropped_Masks"
    image_stem = fn.stem
    mask_files = [f for f in mask_folder.glob(f"{image_stem}_mask*")]
    return mask_files

def extract_external_id(filename):
    """
    Extracts the external_id from the filename.
    """
    return str(filename.stem.split('_')[1])
    
def map_mayo_labels(labels):
    """
    Maps tumor labels to 'Malignant' if it's 'Malignant', otherwise 'Benign'.
    """
    label_map = {"Malignant": "malignant", "Benign": "benign", "Elevated Risk": "benign", "Unknown": "unknown"}
    return [label_map[label] for label in labels]
    
def mayo_dataset_make(dv):
    global mayo_dataset
    mayo_dataset = Path(dv + "/BUS Project Home/Datasets/Mayo_Datasets/mayo_dataset")
    
    # Read annotations_histology.csv
    annotations_df = pd.read_csv(mayo_dataset / "annotations_histology.csv")
    annotations_df['external_id'] = annotations_df['external_id'].astype(str)
    
    mayo_all_images = [f for f in mayo_dataset.glob("Cropped_Grayscale/*.png")]
    
    # Extract external_ids from filenames
    external_ids = [extract_external_id(f) for f in mayo_all_images]
    mismatched_ids = [ext_id for ext_id in external_ids if ext_id not in annotations_df['external_id'].values]
    
    # Lookup tumor labels based on external_ids
    tumor_labels = [annotations_df.loc[annotations_df['external_id'] == ext_id, 'pathology'].values[0] for ext_id in external_ids]
    
    mayo_all_masks = [mayo_get_label(f) for f in mayo_all_images]

    out_df = pd.DataFrame(
        {
            "dataset": ["Mayo_Dataset"] * len(mayo_all_images),
            "image": mayo_all_images,
            "mask": mayo_all_masks,
            "tumor": map_mayo_labels(tumor_labels)
        }
    )
    
    out_df = out_df[out_df['tumor'] != "unknown"]
    
    return out_df

# BUV dataset


def buv_dataset_make(dv):
    buv_dataset = Path(dv + "/BUS Project Home/Datasets/BUV_dataset")

    buv_df = pd.DataFrame(
        {
            "dataset": [],
            "class": [],
            "video": [],
            "image": [],
        }
    )

    for image_class in ["benign", "malignant"]:
        buv_folders = [v for v in (buv_dataset / "rawframes" / image_class).glob("*")]
        buv_folders.sort()
        for v in buv_folders:
            buv_images = [
                f for f in (buv_dataset / "rawframes" / image_class / v.stem).glob("*")
            ]
            buv_images.sort()
            buv_df_short = pd.DataFrame(
                {
                    "dataset": ["BUV_dataset"] * len(buv_images),
                    "class": [image_class] * len(buv_images),
                    "video": [v.stem] * len(buv_images),
                    "image": buv_images,
                }
            )
            buv_df = pd.concat([buv_df, buv_df_short], ignore_index=True)

    buv_df.reset_index(drop=True, inplace=True)

    buv_df["str_index"] = buv_df.index.to_list()
    dataset_length = len(str(buv_df.shape[0]))
    buv_df["str_index"] = buv_df["str_index"].apply(
        lambda x: str(x).rjust(dataset_length, "0")
    )
    return buv_df


# BUSBRA dataset

def busbra_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    label_name = busbra_dataset / "Masks" / f"{fn.stem.replace('bus', 'mask')}.png" 
    return [label_name]


def busbra_dataset_make(dv):
    global busbra_dataset
    busbra_dataset = Path(dv + "/BUS Project Home/Datasets/BUSBRA")

    busbra_labels = pd.read_csv(busbra_dataset / "bus_data.csv")

    busbra_all_images = [f for f in (busbra_dataset / "Images").glob("*.png")]
    busbra_all_masks = [busbra_get_label(f) for f in busbra_all_images]

    # Creating a DataFrame to store the dataset
    busbra_df = pd.DataFrame({
        "dataset": ["BUSBRA"] * len(busbra_all_images),
        "image": busbra_all_images,
        "mask": busbra_all_masks
    })

    busbra_df["tumor"] = [busbra_labels.loc[busbra_labels['ID'] == str(img.stem), 'Pathology'].values[0] if img.stem in busbra_labels['ID'].values else None for img in busbra_all_images]

    return busbra_df
    
    
# Breast Lesions USG

def breast_lesions_get_label(fn):
    """
    Given a path to an image file, returns the path to the mask.
    """
    mask_folder = breast_lesions_dataset / "Masks"
    mask_filename = fn.stem + "_tumor.png"
    mask_file = mask_folder / mask_filename
    if mask_file.exists():
        return [mask_file]
    else:
        return [None]


def breast_lesions_dataset_make(dv):
    global breast_lesions_dataset
    breast_lesions_dataset = Path(dv + "/BUS Project Home/Datasets/Breast_Lesions_USG")
    
    image_files = [f for f in (breast_lesions_dataset / "Original").glob("*.png")]
    mask_files = [breast_lesions_get_label(f) for f in image_files]
    out_df = pd.DataFrame({"image": image_files, "mask": mask_files})
    
    clinical_data = pd.read_excel(
        breast_lesions_dataset / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    )
    clinical_data.rename(columns={"Image_filename": "image", "Classification": "tumor"}, inplace=True)
    
    # Extract filenames from the paths in the "image" column
    out_df["image_filename"] = out_df["image"].apply(lambda x: x.name)
    
    # Merge dataframes based on the filenames, drop NaN values, add the "dataset" column with the value "Breast_Lesions_USG"
    out_df = pd.merge(out_df, clinical_data[["image", "tumor"]], left_on="image_filename", right_on="image", how="left")
    out_df.dropna(subset=["mask"], inplace=True)
    out_df.rename(columns={"image_x": "image"}, inplace=True) #rename after merge
    out_df["dataset"] = "Breast_Lesions_USG"

    # Reorder columns and drop redundant columns
    out_df = out_df[["dataset", "image", "mask", "tumor"]]
    out_df = out_df[out_df['tumor'] != 'normal']

    out_df.reset_index(drop=True, inplace=True)

    return out_df


# Mapping classes to/from pixel values


def labels_ids_bus(multiclass=True):
    """
    Generate mappings between labels and IDs
    """
    if multiclass:
        id2label = {0: "unlabeled", 1: "benign", 2: "malignant"}
        label2id = {"unlabeled": 0, "benign": 1, "malignant": 2}
        num_labels = len(id2label)
    else:
        id2label = {0: "unlabeled", 1: "lesion"}
        label2id = {"unlabeled": 0, "lesion": 1}
        num_labels = len(id2label)
    return id2label, label2id, num_labels