
# scripts/overfit_zbin_t999.py
#
# Debug/diagnostic: train ONLY at a fixed timestep t=999 (maximum noise),
# and validate by DDIM sampling from the SAME fixed noise.
#
# Why this script exists:
# - If DDPM training "looks good" by loss but sampling is pure noise,
#   common causes are: wrong sampling code, wrong scheduler, wrong scaling,
#   wrong conditioning usage, or dataset normalization mismatch.
#
# This script forces a very simple target:
#   Given x_t at t=999, predict the exact noise epsilon.
# If the model can overfit this, sampling from that same fixed noise should
# produce a stable "blurry chest" shape (not perfect), proving the pipeline works.
#
# Outputs:
#   outputs/debug/overfit_zbin_t999/
#     real_*.png
#     recon_stepXXXX_*.png   (DDIM recon samples from fixed noise)
#
# Run:
#   source .venv/bin/activate
#   python -m scripts.overfit_zbin_t999

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from diffusers.training_utils import EMAModel


# =====================
# CONFIG
# =====================
DATA_DIR = "data/raw/slices"
MANIFEST = "data/processed/manifest_zbins.txt"

OUT_DIR = "outputs/debug/overfit_zbin_t999"

IMAGE_SIZE = 256
BATCH_SIZE = 4
LR = 2e-4
STEPS = 3000

SEED = 12345
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FIXED_T = 999          # train only at max noise
DDIM_STEPS = 50        # sampling steps for recon check
DDIM_ETA = 0.0         # deterministic
SAMPLE_EVERY = 500     # save recon at these steps


# =====================
# UTILS
# =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_minus1_plus1(x01: torch.Tensor) -> torch.Tensor:
    return x01 * 2.0 - 1.0


def load_manifest_pairs(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"Manifest not found: {path}")
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fn, zbin = line.split("\t")
            zbin = int(zbin)
            pairs.append((fn, zbin))
    if len(pairs) == 0:
        raise RuntimeError("Manifest is empty.")
    return pairs


# =====================
# DATASET
# =====================
class ZBinSlicesDataset(Dataset):
    def __init__(self, data_dir: str, manifest_path: str, image_size: int = 256):
        self.data_dir = data_dir
        self.image_size = image_size
        self.items = load_manifest_pairs(manifest_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        fn, zbin = self.items[idx]
        path = os.path.join(self.data_dir, fn)
        arr = np.load(path).astype(np.float32)  # expected [0,1]
        x = torch.from_numpy(arr)[None, ...]    # (1,H,W)

        x = x.unsqueeze(0)  # (1,1,H,W)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)    # (1,256,256)

        x = torch.clamp(x, 0.0, 1.0)
        x = to_minus1_plus1(x)

        return {
            "pixel_values": x,
            "class_labels": torch.tensor(zbin).long()
        }


# =====================
# FIXED-NOISE DDIM CHECK
# =====================
@torch.no_grad()
def ddim_recon_from_fixed_noise(
    unet: UNet2DModel,
    ema: EMAModel,
    ddim: DDIMScheduler,
    zbin: int,
    fixed_noise: torch.Tensor,
    steps: int = 50,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Start from fixed_noise and run DDIM sampling conditioned on zbin.
    """
    ema.store(unet.parameters())
    ema.copy_to(unet.parameters())

    x = fixed_noise.clone()
    b = x.size(0)

    ddim.set_timesteps(steps, device=DEVICE)
    for t in ddim.timesteps:
        eps = unet(
            x, t,
            class_labels=torch.full((b,), zbin, device=DEVICE, dtype=torch.long)
        ).sample
        x = ddim.step(eps, t, x, eta=eta).prev_sample

    x = torch.clamp(x, -1.0, 1.0)

    ema.restore(unet.parameters())
    return x


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # DATA
    ds = ZBinSlicesDataset(DATA_DIR, MANIFEST, image_size=IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    batch = next(iter(dl))

    x0 = batch["pixel_values"].to(DEVICE)       # (B,1,256,256) in [-1,1]
    zc = batch["class_labels"].to(DEVICE)       # (B,) 0..3

    print("x0 range:", float(x0.min()), float(x0.max()), "dtype:", x0.dtype)
    print("zc:", zc.tolist(), "dtype:", zc.dtype)
    print(f"Training ONLY on t={FIXED_T} ...")

    # MODEL
    unet = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=4,
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    ema = EMAModel(unet.parameters(), decay=0.999)
    ema.to(DEVICE)

    # FIXED noise / fixed t for training + debugging
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    eps_fixed = torch.randn_like(x0, generator=gen)  # fixed epsilon
    t_fixed = torch.full((x0.size(0),), FIXED_T, device=DEVICE, dtype=torch.long)

    # Create x_t once from fixed eps and fixed t
    x_t_fixed = noise_scheduler.add_noise(x0, eps_fixed, t_fixed)

    # Save real reference
    save_image((x0 + 1) / 2, os.path.join(OUT_DIR, f"real_batch.png"), nrow=2)

    t0 = time.time()

    for step in range(1, STEPS + 1):
        unet.train()

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = unet(x_t_fixed, t_fixed, class_labels=zc).sample
            loss = torch.mean((pred - eps_fixed) ** 2)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ema.step(unet.parameters())

        if step % 100 == 0:
            print(f"[{step}/{STEPS}] loss={loss.item():.6f}")

        # recon check every 500 steps
        if step % SAMPLE_EVERY == 0:
            unet.eval()
            with torch.no_grad():
                # recon-check: eps_pred should match eps_fixed
                mse_eps = torch.mean((pred.detach() - eps_fixed) ** 2).item()
                print(f"    recon check: mse(eps_pred, eps_fixed)={mse_eps:.6f}")

                # DDIM recon from the SAME fixed noise (x starts from eps_fixed)
                # Use per-sample zbin. We'll just run per zbin group:
                for zb in [0, 3]:
                    # fixed_noise batch for recon: start from pure noise
                    fixed_noise = eps_fixed.clone()
                    recon = ddim_recon_from_fixed_noise(
                        unet=unet,
                        ema=ema,
                        ddim=ddim_scheduler,
                        zbin=zb,
                        fixed_noise=fixed_noise,
                        steps=DDIM_STEPS,
                        eta=DDIM_ETA
                    )
                    out = os.path.join(OUT_DIR, f"recon_step{step}_zbin{zb}.png")
                    save_image((recon + 1) / 2, out, nrow=2)

    elapsed = time.time() - t0
    print(f"DONE. elapsed={elapsed:.1f}s")
    print(f"Check: {OUT_DIR}")


if __name__ == "__main__":
    main()
