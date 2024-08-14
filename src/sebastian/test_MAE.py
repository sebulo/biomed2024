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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def test():
    # Parameters
    batch_size = 8
    num_layers = 3
    width = 256
    num_head = 4
    mask_ratio = 0.8

    # Load dataset
    data_dir = "/work3/rapa/challenge_data/test"
    # train_list = "test_files_200.txt"
    train_list = "test_files_200.txt"
    result_dir = "/zhome/28/e/143966/ssr/biomed2024/src/results"
    train_id_list_file = os.path.join(result_dir, train_list)
    train_ids = np.loadtxt(str(train_id_list_file), delimiter=",", dtype=str)
    # Initialize dataset
    dataset = VertebraDataset(
        data_dir=data_dir,
        file_list=train_ids,
        data_type='test'  # or 'mesh' or 'segmentation'
        # data_type='tr'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=False, drop_last=False, num_workers=4)


    model = TR(num_layers, width, num_head, mask_ratio);
    model.load_state_dict(torch.load('../results/mae_model_0.8000.pth'))
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device);
    model.eval()
    probs = []
    labels = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            image, vtx, seg, label = batch
            image, vtx, seg, label = image.to(device), vtx.to(device), seg.to(device), label.to(device)
            emb, mask, loss, logit = model(image, vtx, seg, label)
            probs.append(torch.softmax(logit, dim=-1)[:,-1].detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    print(probs.shape)
    print(labels.shape)
    np.save('prob.npy', probs)
    preds = probs > 0.5;
    print(preds)
    print(accuracy_score(labels, preds), precision_score(labels, preds), recall_score(labels, preds), f1_score(labels, preds))

if __name__ == '__main__':
    test()