import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from VetebraDataset import VertebraDataset
from sklearn.metrics import accuracy_score, f1_score


# Define the ALBEF model
class TR(nn.Module):
    def __init__(self, num_layers, width, num_head, mask_ratio):
        super().__init__()
        # embeddings
        self.width = width
        self.num_layers = num_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=num_head, batch_first=True)
        scale = width**-0.5;
        self.mask_ratio = mask_ratio
        self.token = nn.Parameter(scale * torch.randn(width))

        ## Image
        self.image_embder = nn.Conv3d(1, width, 64, 64); # -> 16x16 patches
        self.image_context_length = 27
        self.image_pos = nn.Parameter(torch.empty(self.image_context_length, width))
        nn.init.normal_(self.image_pos, std=0.01)
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(width)

        ## vtx
        self.vtx_embder = nn.Sequential(nn.Linear(3, width),
                                        nn.SiLU(),
                                        nn.Linear(width, width)
                                        ); # -> 16x16 patches
        self.vtx_context_length = 2
        self.vtx_pos = nn.Parameter(torch.empty(self.vtx_context_length, width))
        nn.init.normal_(self.vtx_pos, std=0.01)      

        ## seg
        self.seg_embder = nn.Conv3d(1, width, 64, 64); # -> 16x16 patches
        self.seg_context_length = 27
        self.seg_pos = nn.Parameter(torch.empty(self.seg_context_length, width))
        nn.init.normal_(self.seg_pos, std=0.01)   

        decoder_layer = nn.TransformerDecoderLayer(d_model=width, nhead=num_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.classifier = nn.Linear(width, 2);

    def forward(self, image, vtx, seg, label, **kwargs):
        # tokenize images
        image = self.image_embder(image).flatten(-3).permute(0,2,1); # B, N, C
        image = image + self.image_pos.unsqueeze(0);
        vtx = self.vtx_embder(vtx); # B, N, C
        vtx = torch.cat((torch.max(vtx, dim=1)[0].unsqueeze(1), torch.mean(vtx, dim=1).unsqueeze(1)), dim=1);
        vtx = vtx + self.vtx_pos.unsqueeze(0);
        seg = self.seg_embder(seg).flatten(-3).permute(0,2,1); # B, N, C
        seg = seg + self.seg_pos.unsqueeze(0);
        ori = self.ln(torch.cat((image, vtx, seg), dim=1));
        if self.training:
            # apply random mask
            mask = torch.rand_like(ori[...,0]);
            mask = mask <= self.mask_ratio;
            emb = mask.unsqueeze(-1) * ori;
        else:
            emb = ori
            mask = None
        emb = torch.cat((self.token.unsqueeze(0) + torch.zeros(emb.shape[0], 1, emb.shape[-1], dtype=emb.dtype, device=emb.device), emb), dim=1);
        emb = self.tr(emb)
        rec = self.decoder(ori.clone(), emb[:,1:,:]);
        rec = self.ln(rec)
        # mlm loss 
        loss_mlm = ((rec[:,:,:] - ori)**2).flatten().mean();
        logit = self.classifier(emb[:,0,...])
        loss_cls = nn.CrossEntropyLoss()(logit, label.long());
        loss = 0.5*loss_mlm + loss_cls;
        return emb, mask, loss, logit


def train():
    # Parameters
    learning_rate = 1e-3
    batch_size = 8
    num_epochs = 200
    num_layers = 3
    width = 256
    num_head = 4
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
        data_type='tr'  # or 'mesh' or 'segmentation'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = VertebraDataset(
        data_dir=data_dir,
        file_list=val_ids,
        data_type='tr'  # or 'mesh' or 'segmentation'
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = TR(num_layers, width, num_head, mask_ratio);
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()
    model = model.to(device)
    best_f1 = 0;
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader):
            model.train();
            optimizer.zero_grad()
            image, vtx, seg, label = batch
            image, vtx, seg, label = image.to(device), vtx.to(device), seg.to(device), label.to(device)
            emb, mask, loss, _ = model(image, vtx, seg, label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        scheduler.step();
        
        print(f"Epoch {epoch + 1}, train Loss: {total_loss / len(dataloader.dataset)}")

        labels = []
        preds = []
        for batch in tqdm(testloader):
            model.eval();
            with torch.no_grad():
                image, vtx, seg, label = batch
                image, vtx, seg, label = image.to(device), vtx.to(device), seg.to(device), label.to(device)
                emb, mask, loss, logit = model(image, vtx, seg, label)
                labels.append(label.detach().cpu().numpy())
                preds.append(torch.argmax(logit, dim=1).detach().cpu().numpy())
        
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        print(f"Epoch {epoch + 1}, acc: {acc:.4f}, f1: {f1:.4f}")


        if f1 > best_f1:
            best_f1 = f1;
            # Save the model
            model_path = os.path.join(result_dir, f"mae_model_{f1:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()