import fusion_utilities as fusion
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import cv2
import torch

class CustomDataset(Dataset):
    def __init__(self, df, 
                 fusion_mode='InputA', target_size=(224,224),
                 hist_eq = False, transform=None):
        
        self.df = df.copy()
        self.fusion_mode = fusion_mode
        self.hist_eq = hist_eq
        self.target_size = target_size
        self.ToTensor = transforms.ToTensor()
        self.transform = transform  # additional transforms

        # Filter out entries from df_labels where the label would be -1
        class_mapping = {'benign': 0, 'malignant': 1}
        self.df['numerical_label'] = self.df['tumor'].map(class_mapping).fillna(-1)
        self.df = self.df[self.df['numerical_label'] != -1]  # Filter out entries with numerical_label = -1
        self.images = self.df['image']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.df.iloc[idx]['image']
        img = cv2.imread(img_path,0) # the 0 makes it read as grayscale

        if self.hist_eq:
            img = cv2.equalizeHist(img)
        
        mask_path = self.df.iloc[idx]['mask']
        mask = cv2.imread(mask_path,0) # reads as grayscale

        if self.fusion_mode == 'InputA':
            img,_= fusion.makeInputA(img,mask,self.target_size)
        elif self.fusion_mode == 'InputB':
            img,_= fusion.makeInputB(img,mask,self.target_size)
        elif self.fusion_mode == 'InputC':
            img,_= fusion.makeInputC(img,mask,self.target_size)
        elif self.fusion_mode == 'InputD':
            img,_,_= fusion.makeInputD(img,mask,self.target_size)
        elif self.fusion_mode == 'InputE':
            img,_= fusion.makeInputE(img,mask,self.target_size)
        elif self.fusion_mode == 'InputF':
            img,_= fusion.makeInputF(img,mask,self.target_size)
        elif self.fusion_mode == 'InputG':
            img,_= fusion.makeInputG(img,mask,self.target_size)
        else:
            raise ValueError("Invalid fusion_mode.")

        # convert to 3 channels if necessary
        img = fusion.convert_to_3_channels(img)
        
        # finally convert to tensor
        final_image = self.ToTensor(img)

        # apply augmentation transforms
        if self.transform:
            final_image = self.transform(final_image)

        label = self.df.iloc[idx]['numerical_label']
        label = torch.tensor(label, dtype=torch.int64)  # Convert label to LongTensor
        external_id = self.df.iloc[idx]['external_id']
        pathology = self.df.iloc[idx]['tumor']
        
        return final_image, label, external_id, pathology