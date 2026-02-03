# scripts/train_diffusers_zbin.py
#
# This is the "best-performing" training script you used:
# - diffusers UNet2DModel with 4 z-bin classes (0..3)
# - cosine-like schedule: squaredcos_cap_v2
# - EMA (diffusers.training_utils.EMAModel)
# - fixed-noise DDIM sampling with eta (helps reduce stuck artifacts)
#
# Outputs:
#   outputs/runs/diffusers_zbin_256/
#     ckpt_stepXXXX.pt
#     samples_fixednoise_seed999_eta0.3_zbin0_stepXXXX.png
#     ...
#     samples_fixednoise_seed999_eta0.3_zbin3_stepXXXX.png
# and also deterministic versions if you set eta=0.0 (in sampling)
#
# IMPORTANT:
# - This script supports resume-from-latest checkpoint automatically.
# - EMA is kept on GPU after resume to avoid cpu/cuda mismatch.

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
MANIFEST = "data/processed/manifest_zbins.txt"   # filename<TAB>zbin
RUN_DIR = "outputs/runs/diffusers_zbin_256"

IMAGE_SIZE = 256
BATCH_SIZE = 8
LR = 2e-4
TOTAL_STEPS = 40000
SAVE_EVERY = 2000
NUM_WORKERS = 2

SEED = 999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DDIM sampling params
SAMPLE_STEPS = 50
SAMPLE_ETA = 0.3  # 0.3 used to reduce "stuck" artifacts (your run). 0.0 deterministic.


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
            parts = line.split("\t")
            if len(parts) != 2:
                raise RuntimeError(f"Bad manifest line: {line}")
            fn = parts[0]
            zbin = int(parts[1])
            if zbin < 0 or zbin > 3:
                raise RuntimeError(f"zbin out of range in manifest: {line}")
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
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        arr = np.load(path).astype(np.float32)  # expected [0,1]
        x = torch.from_numpy(arr)[None, ...]    # (1,H,W)

        x = x.unsqueeze(0)  # (1,1,H,W)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)    # (1,256,256)

        x = torch.clamp(x, 0.0, 1.0)
        x = to_minus1_plus1(x)

        return {
            "pixel_values": x,                        # (1,256,256) in [-1,1]
            "class_labels": torch.tensor(zbin).long() # scalar
        }


# =====================
# FIXED-NOISE DDIM SAMPLING
# =====================
@torch.no_grad()
def sample_fixednoise_ddim(
    unet: UNet2DModel,
    ema: EMAModel,
    ddim: DDIMScheduler,
    zbin: int,
    batch_size: int = 16,
    seed: int = 999,
    eta: float = 0.3,
    steps: int = 50
) -> torch.Tensor:
    """
    Fixed-seed DDIM sampling from EMA weights.
    Returns (B,1,H,W) in [-1,1]
    """
    # apply EMA shadow weights
    ema.store(unet.parameters())
    ema.copy_to(unet.parameters())

    gen = torch.Generator(device=DEVICE).manual_seed(seed + 10 * zbin)
    x = torch.randn((batch_size, 1, IMAGE_SIZE, IMAGE_SIZE), generator=gen, device=DEVICE)

    ddim.set_timesteps(steps, device=DEVICE)
    for t in ddim.timesteps:
        model_out = unet(x, t, class_labels=torch.full((batch_size,), zbin, device=DEVICE, dtype=torch.long)).sample
        x = ddim.step(model_out, t, x, eta=eta).prev_sample

    x = torch.clamp(x, -1.0, 1.0)

    # restore original (non-EMA) weights
    ema.restore(unet.parameters())
    return x


# =====================
# CHECKPOINT IO
# =====================
def save_ckpt(path: str, step: int, unet: UNet2DModel, opt: torch.optim.Optimizer, ema: EMAModel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "unet": unet.state_dict(),
            "opt": opt.state_dict(),
            "ema": ema.state_dict(),
        },
        path
    )


def load_ckpt(path: str, unet: UNet2DModel, opt: torch.optim.Optimizer, ema: EMAModel):
    state = torch.load(path, map_location="cpu")
    step = int(state["step"])
    unet.load_state_dict(state["unet"])
    opt.load_state_dict(state["opt"])
    ema.load_state_dict(state["ema"])
    return step


def move_optimizer_to_device(opt: torch.optim.Optimizer, device: str):
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


# =====================
# MAIN
# =====================
def main():
    set_seed(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)

    # DATA
    ds = ZBinSlicesDataset(DATA_DIR, MANIFEST, image_size=IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    dl_iter = iter(dl)

    # MODEL
    unet = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=4,  # zbin 0..3
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    ema = EMAModel(unet.parameters(), decay=0.999)
    ema.to(DEVICE)

    # RESUME (latest)
    step = 0
    ckpts = [f for f in os.listdir(RUN_DIR) if f.startswith("ckpt_step") and f.endswith(".pt")]
    if ckpts:
        ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("step")[1].split(".pt")[0]))
        last_ckpt = os.path.join(RUN_DIR, ckpts_sorted[-1])
        print("Resuming from:", last_ckpt)
        step = load_ckpt(last_ckpt, unet, opt, ema)
        move_optimizer_to_device(opt, DEVICE)
        ema.to(DEVICE)  # CRITICAL (avoid cpu/cuda mismatch)
    else:
        print("No checkpoint found. Starting from scratch.")

    t0 = time.time()

    # TRAIN LOOP
    while step < TOTAL_STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        step += 1

        x0 = batch["pixel_values"].to(DEVICE)                  # (B,1,H,W) [-1,1]
        labels = batch["class_labels"].to(DEVICE).view(-1)     # (B,) long

        noise = torch.randn_like(x0)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x0.size(0),), device=DEVICE).long()
        x_t = noise_scheduler.add_noise(x0, noise, t)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = unet(x_t, t, class_labels=labels).sample
            loss = torch.mean((pred - noise) ** 2)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # EMA update
        ema.step(unet.parameters())

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"[{step}/{TOTAL_STEPS}] loss={loss.item():.6f} elapsed={elapsed:.1f}s")

        # SAVE samples + ckpt
        if step % SAVE_EVERY == 0:
            unet.eval()
            with torch.no_grad():
                for zb in range(4):
                    samples = sample_fixednoise_ddim(
                        unet=unet,
                        ema=ema,
                        ddim=ddim_scheduler,
                        zbin=zb,
                        batch_size=16,
                        seed=SEED,
                        eta=SAMPLE_ETA,
                        steps=SAMPLE_STEPS
                    )
                    out_png = os.path.join(
                        RUN_DIR,
                        f"samples_fixednoise_seed{SEED}_eta{SAMPLE_ETA}_zbin{zb}_step{step}.png"
                    )
                    save_image((samples + 1) / 2, out_png, nrow=4)

            ckpt_path = os.path.join(RUN_DIR, f"ckpt_step{step}.pt")
            save_ckpt(ckpt_path, step, unet, opt, ema)
            unet.train()

    # final save
    ckpt_path = os.path.join(RUN_DIR, f"ckpt_step{step}.pt")
    save_ckpt(ckpt_path, step, unet, opt, ema)

    print("DONE. Last step:", step)


if __name__ == "__main__":
    main()

