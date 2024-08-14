import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
# from dtu_spine_config import DTUConfig
from train_MAE import TR
from VetebraDataset import VertebraDataset

def test():
    # Parameters
    learning_rate = 1e-3
    batch_size = 4
    num_epochs = 50
    num_layers = 3
    width = 128
    num_head = 2
    mask_ratio = 0.6

    # Load dataset
    data_dir = "/work3/rapa/challenge_data/train"
    train_list = "custom_train_list_100.txt"
    result_dir = "/zhome/28/e/143966/ssr/biomed2024/src/results"
    train_id_list_file = os.path.join(result_dir, train_list)
    train_ids = np.loadtxt(str(train_id_list_file), delimiter=",", dtype=str)

    data_dir = "/work3/rapa/challenge_data/train"
    val_list = "custom_train_list_100.txt"
    val_id_list_file = os.path.join(result_dir, val_list)
    val_ids = np.loadtxt(str(val_id_list_file), delimiter=",", dtype=str)
    # Initialize dataset
    dataset = VertebraDataset(
        data_dir=data_dir,
        file_list=train_ids,
        data_type='test'  # or 'mesh' or 'segmentation'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8)


    model = TR(num_layers, width, num_head, mask_ratio);
    model.load_state_dict(torch.load('../results/mae_model.pth'))
    print(model)

    model.eval()
    probs = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            image, vtx, seg, label = batch
            emb, mask, loss, logit = model(image, vtx, seg, label)
            probs.append(torch.softmax(logit, dim=-1)[:,-1].detach().cpu().numpy())
    probs = np.concatenate(probs)
    print(probs.shape)
    np.save('prob.npy', probs)


if __name__ == '__main__':
    test()