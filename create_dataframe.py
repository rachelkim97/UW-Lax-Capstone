import pandas as pd
from sklearn.model_selection import train_test_split

from bus_data import *

def create_dataframe(wd_path):
    all_data_df = pd.DataFrame(
        {
            "dataset": [],
            "image": [],
            "mask": [],
            "tumor": [],
        }
    )

    all_data_df = pd.concat([all_data_df, busis_dataset_make(wd_path)], ignore_index=True) # BUSIS
    all_data_df = pd.concat([all_data_df, bus_dataset_b_make(wd_path)], ignore_index=True) # BUS Dataset B
    all_data_df = pd.concat([all_data_df, dataset_busi_with_gt_make(wd_path)], ignore_index=True) # Dataset BUSI with GT
    all_data_df = pd.concat([all_data_df, mayo_dataset_make(wd_path)], ignore_index=True)  # Mayo Dataset
    all_data_df = pd.concat([all_data_df, busbra_dataset_make(wd_path)], ignore_index=True) # BUSBRA
    all_data_df = pd.concat([all_data_df, breast_lesions_dataset_make(wd_path)], ignore_index=True) # BrEaSt Lesions USG

    # Function to extract external ID based on dataset name
    def extract_external_id(row):
        ending_parts = str(row['image']).split('/')[-1]
        dataset = row['dataset']
        if dataset == 'BUSIS':
            external_id = int(ending_parts.split('.png')[0].replace('case', ''))
        elif dataset == 'BUS_Dataset_B':
            external_id = int(ending_parts.split('.png')[0])
        elif dataset == 'Dataset_BUSI_with_GT':
            external_id = int(ending_parts.split(')')[0].split('(')[1])
        elif dataset == 'Mayo_Dataset':
            external_id = int(ending_parts.split('_')[1]) # "ex: cropped_00000_long.png"
        elif dataset == 'BUSBRA':
            external_id = int(ending_parts.split('_')[1].split('-')[0])
        elif dataset == 'Breast_Lesions_USG':
            external_id = int(ending_parts.split('.png')[0].replace('case',''))
        return external_id

    # Apply the lambda function to create the 'external_id' column
    all_data_df['external_id'] = all_data_df.apply(lambda row: extract_external_id(row), axis=1)
    all_data_df['grouped_ids'] = all_data_df['dataset'] + '_' + all_data_df['external_id'].astype(str)
    all_data_df = all_data_df.astype(str)

    # Convert mask paths to correct format to be used in CustomDataset
    def extract_path_part(path):
        if isinstance(path, list) and path:  # Check if it's a non-empty list
            return str(path[0]).split("'")[1] if str(path[0]).count("'") > 1 else str(path[0])
        else:
            return str(path).split("'")[1] if str(path).count("'") > 1 else str(path)

    all_data_df['mask'] = all_data_df['mask'].apply(extract_path_part)

     # Get unique grouped_ids (patients)
    unique_grouped_ids = all_data_df['grouped_ids'].unique()

    # Split unique grouped_ids into train, val, and test sets
    train_grouped_ids, test_grouped_ids = train_test_split(unique_grouped_ids, test_size=0.3, random_state=42)
    val_grouped_ids, test_grouped_ids = train_test_split(test_grouped_ids, test_size=0.5, random_state=42)

    # Assign each data point to train, val, or test based on grouped_ids
    all_data_df['split'] = 'train'
    all_data_df.loc[all_data_df['grouped_ids'].isin(val_grouped_ids), 'split'] = 'val'
    all_data_df.loc[all_data_df['grouped_ids'].isin(test_grouped_ids), 'split'] = 'test'

    return all_data_df