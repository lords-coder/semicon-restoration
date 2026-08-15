import os
import torch
import numpy as np
import cv2
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.model import RestoreNet


def evaluate(input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = RestoreNet().to(device)
    checkpoint = torch.load('models/checkpoint_epoch300.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    os.makedirs(output_dir, exist_ok=True)

    total_time = 0

    for f in files:
        lr = np.load(os.path.join(input_dir, f)).astype(np.float32) / 255.0
        lr_tensor = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

        start = time.time()
        with torch.no_grad():
            output = model(lr_tensor)
        total_time += time.time() - start

        output = output.squeeze().cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f.replace('.npy', '.png')), output)

    print(f"Processed {len(files)} images")
    print(f"Average time: {total_time/len(files)*1000:.2f} ms per image")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    evaluate(args.input_dir, args.output_dir)