import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig
from PIL import Image

from torch.utils.data import DataLoader
from VetebraDataset import VertebraDataset


# Function to fine-tune the VGG model
def fine_tune_vgg(settings):
    # Parameters
    data_dir = settings["data_dir"]
    train_list = settings["data_set"]
    result_dir = settings["result_dir"]
    num_classes = 2  # Assuming binary classification: normal or outlier
    learning_rate = 1e-4
    batch_size = 16
    num_epochs = 10

    # Define transformations for the training data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    train_id_list_file = os.path.join(result_dir, train_list)
    train_ids = np.loadtxt(str(train_id_list_file), delimiter=",", dtype=str)
    # Initialize dataset
    dataset = VertebraDataset(
        data_dir=data_dir,
        file_list=train_ids,
        transform=transform,
        data_type='image'  # or 'mesh' or 'segmentation'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load pre-trained VGG model
    model = models.vgg16(pretrained=True)

    # Modify the classifier to fit the number of classes in your dataset
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Fine-tune the model
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images in dataloader:
            # Assuming binary labels are either 0 or 1
            labels = torch.zeros(images.size(0), dtype=torch.long)  # Replace with your actual labels
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    # Save the fine-tuned model
    model_path = os.path.join(result_dir, "vgg_finetuned.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train-finetune-vgg')
    config = DTUConfig(args)
    if config.settings is not None:
        fine_tune_vgg(config.settings)