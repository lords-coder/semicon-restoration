import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.dataset import KlaDataset
from models.model import RestoreNet


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")

    train_dataset = KlaDataset(
        gt_dir='C:/hackathon_project/train_raw/train/GT',
        lr_dir='C:/hackathon_project/train_raw/train/NoisyLR',
        augment=True
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

    model = RestoreNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.L1Loss()

    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print("Starting training...")

    for epoch in range(300, 1001):
        model.train()
        total_loss = 0

        for lr_img, gt_img in tqdm(train_loader, desc=f"Epoch {epoch}"):
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)

            optimizer.zero_grad()
            output = model(lr_img)
            loss = criterion(output, gt_img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/checkpoint_epoch{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")

    torch.save(model.state_dict(), 'models/final.pth')
    print("Training done! Model saved.")


if __name__ == "__main__":
    train()