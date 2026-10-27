import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from tqdm import tqdm


# Activations

def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6

def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)

class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


# LayerNorm2d

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# Window Attention

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.num_heads) ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    return x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size ** 2))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

class LiteEnhanceBlockWithShiftedWindow(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.attn = WindowAttention(channels, window_size, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        Hp, Wp = x.shape[2:]

        if self.shift:
            shift_size = ws // 2
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(2, 3))

        windows = window_partition(x, ws)
        windows = self.attn(windows)
        out = window_reverse(windows, ws, Hp, Wp)

        if self.shift:
            shift_size = ws // 2
            out = torch.roll(out, shifts=(shift_size, shift_size), dims=(2, 3))

        out = out[:, :, :H, :W]
        x = x[:, :, :H, :W]

        return out + x


# Conv Blocks

class ConvBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False),
            LayerNorm2d(16),
            HardSwish(),
            nn.Conv2d(16, 32, 1, bias=False),
            LayerNorm2d(32),
            HardSwish()
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False),
            LayerNorm2d(32),
            HardSwish(),
            nn.Conv2d(32, 64, 1, bias=False),
            LayerNorm2d(64),
            HardSwish()
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock3(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            LayerNorm2d(64),
            HardSwish(),
            nn.Conv2d(64, 32, 1, bias=False),
            LayerNorm2d(32),
            HardSwish()
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock4(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw = nn.Conv2d(80, 80, 3, padding=1, groups=80, bias=False)
        self.ln = LayerNorm2d(80)
        self.wa1 = LiteEnhanceBlockWithShiftedWindow(80, shift=False)
        self.wa2 = LiteEnhanceBlockWithShiftedWindow(80, shift=True)
        self.pw = nn.Conv2d(80, 32, 1, bias=False)
        self.lnn = LayerNorm2d(32)
        self.hs = HardSwish()

    def forward(self, x):
        x = self.hs(self.ln(self.dw(x)))
        x = self.wa1(x)
        x = self.wa2(x)
        x = self.hs(self.lnn(self.pw(x)))
        return x


# Network

class Mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(3, 16, 1, bias=False)
        self.output = nn.Conv2d(32, 3, 1, bias=False)
        self.b1 = ConvBlock1()
        self.b2 = ConvBlock2()
        self.b3 = ConvBlock3()
        self.b4 = ConvBlock4()

    def forward(self, x):
        x = self.input(x)
        x1 = self.b1(x)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        x3 = torch.cat([x, x1, x3], dim=1)
        x4 = self.b4(x3)
        return self.output(x4)


# Dataset

class Haze1kDataset(Dataset):
    def __init__(self, root, split="train", crop_size=256):
        self.input_dir = os.path.join(root, split, "input")
        self.target_dir = os.path.join(root, split, "target")
        self.images = sorted(os.listdir(self.input_dir))

        if split == "train":
            self.transform = T.Compose([
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor()
            ])
        else:
            self.transform = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        inp = Image.open(os.path.join(self.input_dir, self.images[idx])).convert("RGB")
        tar = Image.open(os.path.join(self.target_dir, self.images[idx])).convert("RGB")

        seed = torch.seed()
        torch.manual_seed(seed)
        inp = self.transform(inp)
        torch.manual_seed(seed)
        tar = self.transform(tar)

        return inp, tar


# SSIM

def ssim_loss(x, y):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * sigma_xy + 0.03 ** 2)) / \
           ((mu_x ** 2 + mu_y ** 2 + 0.01 ** 2) * (sigma_x + sigma_y + 0.03 ** 2))
    return ((1 - ssim) / 2).clamp(0, 1).mean()

def ssim_metric(x, y):
    return 1 - ssim_loss(x, y)


# Training

def train_model(model, train_loader, val_loader, epochs=350, lr=1e-4, device="cuda"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    l1 = nn.L1Loss()

    best_psnr = -1e9

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inp, tar in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inp, tar = inp.to(device), tar.to(device)
            out = model(inp)

            loss = l1(out, tar) + 0.5 * ssim_loss(out, tar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation 
        model.eval()
        psnr_total, ssim_total = 0.0, 0.0

        with torch.no_grad():
            for inp, tar in val_loader:
                inp, tar = inp.to(device), tar.to(device)
                out = model(inp)

                mse = torch.mean((out - tar) ** 2)
                psnr_total += 10 * torch.log10(1.0 / mse).item()
                ssim_total += ssim_metric(out, tar).item()

        avg_psnr = psnr_total / len(val_loader)
        avg_ssim = ssim_total / len(val_loader)

        print(
            f"Epoch {epoch+1}: "
            f"TrainLoss={train_loss / len(train_loader):.4f}, "
            f"Val_PSNR={avg_psnr:.2f}, "
            f"Val_SSIM={avg_ssim:.4f}"
        )

        # Saving the best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Saved new best model (PSNR={best_psnr:.2f})")

    # Saving the final model
    torch.save(model.state_dict(), "final_model.pth")
    print("Training finished. Final model saved as final_model.pth")




# Main

if __name__ == "__main__":
    root = "./Distributed_haze1k"

    full_train = Haze1kDataset(root, "train")

    n_total = len(full_train)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val

    train_set, val_set = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Mynet().to(device)

    train_model(model, train_loader, val_loader, epochs=350, device=device)