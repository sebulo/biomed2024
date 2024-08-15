import numpy as np
import pandas as pd
from skimage import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision import tv_tensors

import matplotlib.pyplot as plt

import pandas as pd
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def read_file(path):
    # Read the segmentation and turn into a numpy array
    try:
        img = sitk.ReadImage(path)
    except RuntimeError as e:
        print(f"Got an exception {str(e)}")
        print(f"Error reading {path}")
        return None

    return sitk.GetArrayFromImage(img)

class Standardize(object):
    
    def __init__(self):
        self.x = 0

    def __call__(self, image):
        image = image - torch.mean(image, (1,2))[:,None,None]
        image = image / torch.std(image, (1,2))[:,None,None]
        return image
        
class ChallengeDataset(Dataset):

    def __init__(self, df, images, masks, augment=False):
        
        self.df = df
        self.images = images
        self.masks = masks

        if augment:
            self.transforms = v2.Compose([v2.RandomVerticalFlip(p=0.5),
                                          v2.RandomHorizontalFlip(p=0.5),
                                          v2.RandomRotation(degrees=90),
                                          v2.RandomResizedCrop(size=(224,224), scale=(0.5,1.0), ratio=(0.75,1.333333)),
                                          v2.Resize(size=(224,224)),
                                          Standardize()])
        else:
            self.transforms = v2.Compose([v2.Resize(size=(224,224)),
                                          Standardize()])
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):

        normal_idx = np.argwhere(self.df['outlier'] == 0).ravel()
        outlier_idx = np.argwhere(self.df['outlier'] > 0).ravel()

        np.random.shuffle(normal_idx)
        np.random.shuffle(outlier_idx)

        if np.random.rand() > 0.5:
            idx = outlier_idx[0]
        else:
            idx = normal_idx[0]

        # Load image and mask
        image = self.images[idx][:,:,None]
        mask = self.masks[idx][:,:,None]

        image = np.concatenate([image,mask], axis=-1)

        image = np.moveaxis(image, -1, 0)

        # Convert to tv_tensors
        image = tv_tensors.Image(torch.tensor(image))

        # Get targets
        target = torch.tensor([self.df['outlier'].iloc[idx]]) > 0

        # Apply transforms
        image = self.transforms(image)

        # print(image.shape, target.shape)s
        
        sample = (image.to(torch.float32), target.to(torch.float32))

        return sample
        
def get_data_loaders(batch_size=2):
    '''
    Creates pytorch DataLoaders for the train, validation and test sets.
    '''
    
    df = pd.read_csv('train_data.csv')
    
    train_samples, val_samples = train_test_split(np.unique(df['sample']), test_size=0.20, random_state=12345)
    
    train_df = df[df['sample'].isin(train_samples)]
    val_df = df[df['sample'].isin(val_samples)]

    images = np.load('train_pca_images.npy')
    masks = np.load('train_pca_masks.npy')

    train_images = images[df['sample'].isin(train_samples)]
    train_masks = masks[df['sample'].isin(train_samples)]
    
    val_images = images[df['sample'].isin(val_samples)]
    val_masks = masks[df['sample'].isin(val_samples)]
    

    train_dataset = ChallengeDataset(train_df, train_images, train_masks, augment=True)
    val_dataset = ChallengeDataset(val_df, val_images, val_masks, augment=False)
    # train_loader = DataLoader(train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           num_workers=0)
    # val_loader = DataLoader(val_dataset,
    #                         batch_size=batch_size,
    #                         shuffle=False,
    #                         num_workers=0)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16,
                            persistent_workers=True)
    
    return train_loader, val_loader