import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm

DATA_PATH = "data"

class HarmoMedDataset(Dataset):
    def __init__(self, root_dir, size=256):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.files = sorted(os.listdir(self.input_dir))
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        inp = cv2.imread(os.path.join(self.input_dir, fname))
        tar = cv2.imread(os.path.join(self.target_dir, fname))

        inp = cv2.resize(inp, (self.size, self.size))
        tar = cv2.resize(tar, (self.size, self.size))

        inp = inp / 255.0
        tar = tar / 255.0

        inp = torch.tensor(inp).permute(2,0,1).float()
        tar = torch.tensor(tar).permute(2,0,1).float()

        return inp, tar

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), nn.ReLU(),
            nn.Conv2d(64,64,3,1,1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1), nn.ReLU(),
            nn.Conv2d(128,128,3,1,1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128,256,3,1,1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256,128,2,2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1), nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128,64,2,2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1), nn.ReLU()
        )

        self.out = nn.Conv2d(64,3,1)

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2,e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1,e1], dim=1))

        return torch.sigmoid(self.out(d1))

def ssim_loss(img1, img2):
    img1 = img1.detach().cpu().numpy().transpose(0,2,3,1)
    img2 = img2.detach().cpu().numpy().transpose(0,2,3,1)

    scores = []
    for i in range(len(img1)):
        scores.append(
            ssim(img1[i], img2[i], channel_axis=2, data_range=1.0)
        )

    return 1 - np.mean(scores)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HarmoMedDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 30

for epoch in range(EPOCHS):
    total_loss = 0

    for inp, tar in tqdm(loader):
        inp, tar = inp.to(device), tar.to(device)

        out = model(inp)

        l1 = torch.mean(torch.abs(out - tar))
        ssim_l = ssim_loss(out, tar)

        loss = l1 + 0.5 * ssim_l

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(loader):.4f}")