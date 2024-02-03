import os
from pathlib import Path
from loguru import logger
import urllib.request
import zipfile
import shutil
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.data import DataLoader
from PIL import Image
import tqdm
import torch

DATASET_DIR = "./dataset/"
dataset_dir = "./dataset/website_stimuli"
DATASET_LOCATION = os.path.join(DATASET_DIR, "all.csv")
IMAGES_DATASET_LOCATION = os.path.join(DATASET_DIR, "website_stimuli")
DATASET_URL = "http://iis.seas.harvard.edu/resources/aesthetics-chi14/SupplementaryMaterials.zip"
stimuli_url = "http://iis.seas.harvard.edu/resources/aesthetics-chi14/website_stimuli.zip"
#IMAGE_SIZE = (160, 120)
# IMAGE_SIZE of the dataset is roughly ~1024 by ~768, divided by two:
IMAGE_SIZE = (512, 384)

accelerator = Accelerator()


class CustomImageDataset(Dataset):
    def __init__(self, csv_path, image_path):
        logger.info(f"Reading csv from {csv_path}...")
        df = pd.read_csv(csv_path, usecols=["mean_response", "website"], dtype={'mean_response': 'float32'})
        self.mean_responses = df[["mean_response", "website"]].groupby("website").mean()
        if image_path[-1] != "/":
            image_path += "/"
        self.mean_responses["filename"] = [image_path + x.replace("_", "/") + ".png" for x in
                                           self.mean_responses.index.values]
        self.transform = Compose([
            Resize(IMAGE_SIZE),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.mean_responses)

    def __getitem__(self, idx):
        img_path = self.mean_responses.filename.iloc[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.mean_responses.mean_response.iloc[idx].reshape((1,))
        image = self.transform(image)
        logger.trace(f"Image dtype: {image.dtype}")
        logger.trace(f"Label: {label}")
        logger.trace(f"Label Dtype: {label.dtype}")
        return image, label


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=0)
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    if not os.path.isfile(DATASET_LOCATION):
        logger.info("CSV Dataset not found, downloading...")
        download_target_path = os.path.join("/tmp", "SupplementaryMaterials.zip")
        urllib.request.urlretrieve(DATASET_URL, download_target_path)
        with zipfile.ZipFile(download_target_path, 'r') as zip_ref:
            zip_ref.extract(member="SupplementaryMaterials/all.csv", path=DATASET_DIR)
        shutil.move(os.path.join(DATASET_DIR, "SupplementaryMaterials/all.csv"), DATASET_DIR)
        shutil.rmtree(os.path.join(DATASET_DIR, "SupplementaryMaterials/"))
        os.remove(download_target_path)
    if not os.path.isdir(IMAGES_DATASET_LOCATION):
        logger.info("Image Dataset not found, downloading...")
        download_target_path = os.path.join("/tmp", "website_stimuli.zip")
        urllib.request.urlretrieve(stimuli_url, download_target_path)
        with zipfile.ZipFile(download_target_path, 'r') as zip_ref:
            zip_ref.extractall(IMAGES_DATASET_LOCATION)
        os.remove(download_target_path)

    dataset = CustomImageDataset(DATASET_LOCATION, dataset_dir)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = CustomModel().float()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    for epoch in tqdm.tqdm(range(10)):
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            logger.info(f"Loss: {loss}")
            accelerator.backward(loss)
            optimizer.step()
    torch.save(model.state_dict(), "./model.pt")
