import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as T

# import Mynet from your training file
from train import Mynet

# Device

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset

class Haze1kDataset(torch.utils.data.Dataset):
    def __init__(self, root, split):
        self.input_dir = os.path.join(root, split, "input")
        self.target_dir = os.path.join(root, split, "target")
        self.images = sorted(os.listdir(self.input_dir))
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        inp = Image.open(os.path.join(self.input_dir, self.images[idx])).convert("RGB")
        tar = Image.open(os.path.join(self.target_dir, self.images[idx])).convert("RGB")
        return self.transform(inp), self.transform(tar)


# SSIM metric

def ssim_metric(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

    return ssim.mean()


# Evaluation function

def evaluate(model, dataloader):
    model.eval()
    psnr_total = 0.0
    ssim_total = 0.0

    with torch.no_grad():
        for inp, tar in dataloader:
            inp = inp.to(device)
            tar = tar.to(device)

            out = model(inp)

            # PSNR
            mse = torch.mean((out - tar) ** 2)
            psnr = 10 * torch.log10(1.0 / mse)

            # SSIM
            ssim_val = ssim_metric(out, tar)

            psnr_total += psnr.item()
            ssim_total += ssim_val.item()

    avg_psnr = psnr_total / len(dataloader)
    avg_ssim = ssim_total / len(dataloader)

    return avg_psnr, avg_ssim


# Main

def main():
    root = "./Distributed_haze1k"
    
    model_path = "best_model.pth"

    # Load model
    model = Mynet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nEvaluating {model_path}\n")

    for split in ["test_thin", "test_moderate", "test_thick"]:
        dataset = Haze1kDataset(root, split)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        psnr, ssim = evaluate(model, loader)
        print(f"{split:14s} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

if __name__ == "__main__":
    main() 

