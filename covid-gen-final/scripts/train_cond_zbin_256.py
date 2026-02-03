
# scripts/train_cond_zbin_256.py
#
# Z-BIN conditioned diffusion (256x256).
# - z_norm is discretized into 4 bins: zbin = 0..3
# - We train a conditional U-Net with class embeddings (num_class_embeds=4)
# - Sampling saves per-zbin grids so you can compare bins during training.
#
# Outputs:
#   outputs/runs/train_cond_zbin_256/
#     ckpt_stepXXXXX.pt
#     samples_zbin0_stepXXXXX.png
#     samples_zbin1_stepXXXXX.png
#     samples_zbin2_stepXXXXX.png
#     samples_zbin3_stepXXXXX.png

import os
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler


# =====================
# CONFIG
# =====================
DATA_DIR = "data/raw/slices"
MANIFEST = "data/processed/manifest_zbins.txt"  # must exist (lines: filename<TAB>zbin)
OUT_DIR = "outputs/runs/train_cond_zbin_256"

IMAGE_SIZE = 256
BATCH_SIZE = 8
LR = 2e-4
TOTAL_STEPS = 12000
SAVE_EVERY = 2000
NUM_WORKERS = 2

SEED = 999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
            # expected: "coronacases_007_z0088.npy\t3"
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
    """
    Loads (filename, zbin) from manifest.
    Assumes .npy arrays are already in [0,1] (your corrected pipeline).
    """
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
            "class_labels": torch.tensor(zbin).long(),# scalar long
            "filename": fn
        }


# =====================
# SAMPLING
# =====================
@torch.no_grad()
def sample_ddim_fixed(
    unet: UNet2DModel,
    ddim: DDIMScheduler,
    class_id: int,
    batch_size: int = 16,
    image_size: int = 256,
    seed: int = 999,
    eta: float = 0.3,
    steps: int = 50
) -> torch.Tensor:
    """
    Fixed-seed DDIM sampling so samples are comparable across checkpoints.
    eta>0 can help reduce "stuck/frozen" artifacts.
    returns: (B,1,H,W) in [-1,1]
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 10 * class_id)
    x = torch.randn((batch_size, 1, image_size, image_size), generator=gen, device=DEVICE)

    ddim.set_timesteps(steps, device=DEVICE)
    for t in ddim.timesteps:
        model_out = unet(
            x,
            t,
            class_labels=torch.full((batch_size,), class_id, device=DEVICE, dtype=torch.long)
        ).sample
        x = ddim.step(model_out, t, x, eta=eta).prev_sample

    return torch.clamp(x, -1.0, 1.0)


# =====================
# MAIN TRAIN
# =====================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------------
    # DATA
    # ---------------------
    ds = ZBinSlicesDataset(data_dir=DATA_DIR, manifest_path=MANIFEST, image_size=IMAGE_SIZE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    dl_iter = iter(dl)

    # ---------------------
    # MODEL
    # ---------------------
    unet = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=4,  # zbin classes: 0..3
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ---------------------
    # RESUME (latest ckpt)
    # ---------------------
    step = 0
    ckpts = [f for f in os.listdir(OUT_DIR) if f.startswith("ckpt_step") and f.endswith(".pt")]
    if ckpts:
        ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("step")[1].split(".pt")[0]))
        last_ckpt = os.path.join(OUT_DIR, ckpts_sorted[-1])
        print("Resuming from:", last_ckpt)
        state = torch.load(last_ckpt, map_location="cpu")
        step = int(state["step"])
        unet.load_state_dict(state["unet"])
        opt.load_state_dict(state["opt"])
        # move optimizer state tensors to GPU
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(DEVICE)
    else:
        print("No checkpoint found. Starting from scratch.")

    t0 = time.time()

    # ---------------------
    # TRAIN LOOP
    # ---------------------
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

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"[{step}/{TOTAL_STEPS}] loss={loss.item():.6f} elapsed={elapsed:.1f}s")

        # ---------------------
        # SAVE SAMPLES + CKPT
        # ---------------------
        if step % SAVE_EVERY == 0:
            unet.eval()
            with torch.no_grad():
                for zb in range(4):
                    samples = sample_ddim_fixed(
                        unet=unet,
                        ddim=ddim_scheduler,
                        class_id=zb,
                        batch_size=16,
                        image_size=IMAGE_SIZE,
                        seed=SEED,
                        eta=0.3,
                        steps=50
                    )
                    out_png = os.path.join(OUT_DIR, f"samples_zbin{zb}_step{step}.png")
                    save_image((samples + 1) / 2, out_png, nrow=4)

            ckpt_path = os.path.join(OUT_DIR, f"ckpt_step{step}.pt")
            torch.save(
                {
                    "step": step,
                    "unet": unet.state_dict(),
                    "opt": opt.state_dict(),
                },
                ckpt_path
            )
            unet.train()

    print("DONE. Last step:", step)


if __name__ == "__main__":
    main()
