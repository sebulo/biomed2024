import numpy as np
import pandas as pd
from skimage import io
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import v2
from torchvision import tv_tensors

import matplotlib.pyplot as plt

class RandomGaussianBlur(object):
    
    def __init__(self, min_sigma=0, max_sigma=1):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, image):
        sigma = float(self.min_sigma + torch.rand(1) * (self.max_sigma - self.min_sigma))
        kernel_size = int(4 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        image = v2.functional.gaussian_blur(image, (kernel_size,kernel_size), sigma)
        return image
        
class RandomNoise(object):
    
    def __init__(self, min_scale=0, max_scale=0.01):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image):
        scale = float(self.min_scale + torch.rand(1) * (self.max_scale - self.min_scale))
        noise = torch.rand_like(image) * scale - 0.5 * scale
        image = image + noise
        return image

class Standardize(object):
    
    def __init__(self, channel_mean, channel_std):
        self.channel_mean = channel_mean
        self.channel_std = channel_std

    def __call__(self, image):
        image = image - self.channel_mean[:,None,None]
        image = image / self.channel_std[:,None,None]
        return image
        
class ChallengeDataset(Dataset):

    def __init__(self, X, y, augment=False):
        
        self.X = X
        self.y = y

        channel_mean = np.array([1.32242872, 2.42520988, 2.5470593])
        channel_std = np.array([3.06781983, 3.93189598, 3.8842867])

        if augment:
            self.transforms = v2.Compose([v2.RandomVerticalFlip(p=0.5),
                                          v2.RandomHorizontalFlip(p=0.5),
                                          v2.RandomRotation(degrees=90),
                                          v2.RandomResizedCrop(size=(224,224), scale=(0.5,1.0), ratio=(0.75,1.333333)),
                                          v2.RandomChoice([RandomNoise(min_scale=0, max_scale=0.02),
                                                           RandomGaussianBlur(min_sigma=0, max_sigma=2)]),
                                          Standardize(channel_mean=channel_mean, channel_std=channel_std),
                                          v2.Resize(size=(224,224))])
        else:
            self.transforms = v2.Compose([Standardize(channel_mean=channel_mean, channel_std=channel_std),
                                          v2.Resize(size=(224,224))])
        
        self.elastic_transform = v2.Compose([v2.ElasticTransform(alpha=75.0, sigma=5)])
            
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):

        normal_idx = np.argwhere(self.y == 0).ravel()
        outlier_idx = np.argwhere(self.y == 1).ravel()

        np.random.shuffle(normal_idx)
        np.random.shuffle(outlier_idx)

        if np.random.rand() > 0.5:
            idx = outlier_idx[0]
        else:
            idx = normal_idx[0]

        # Load image and mask
        image = self.X[idx]

        image = np.moveaxis(image, -1, 0)

        # Convert to tv_tensors
        image = tv_tensors.Image(torch.tensor(image))

        # Get targets
        target = torch.tensor([self.y[idx]])

        # Apply transforms
        image = self.transforms(image)

        # if target[0] == 0:
        #     if np.random.rand() > 0.5:
        #         # plt.imshow(image.detach().cpu().numpy()[1,:,:])
        #         # plt.show()
        #         image = self.elastic_transform(image)
        #         target[0] == 1
        #         # plt.imshow(image.detach().cpu().numpy()[1,:,:])
        #         # plt.show()
        
        sample = (image.to(torch.float32), target.to(torch.float32))

        return sample
        
def get_data_loaders(batch_size=2):
    '''
    Creates pytorch DataLoaders for the train, validation and test sets.
    '''
    
    X_trainval = np.load('X_train.npy')
    y_trainval = np.load('y_train.npy')

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=12345)
    sss.get_n_splits(X_trainval, y_trainval)
    
    for i, (train_index, val_index) in enumerate(sss.split(X_trainval, y_trainval)):
        if i == 0:
            break

    X_train = X_trainval[train_index]
    y_train = y_trainval[train_index]
    X_val = X_trainval[val_index]
    y_val = y_trainval[val_index]
    
    X_test = np.load('X_test.npy')

    train_dataset = ChallengeDataset(X_train, y_train, augment=True)
    val_dataset = ChallengeDataset(X_val, y_val, augment=False)
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
    
    return train_loader, val_loader, X_test