# scripts/train_cond_256.py
#
# Conditional diffusion (256x256) using diffusers UNet2DModel with 3 discrete "z-classes"
# derived from z_norm in [0,1]. This matches the early project stage where you sampled
# conditions like z=0.1 / 0.5 / 0.9 by binning depth into 3 groups.
#
# Output:
#   outputs/runs/train_cond_256/
#     ckpt_stepXXXXX.pt
#     samples_fixednoise_seed12345_eta0.0_zc0_stepXXXXX.png
#     samples_fixednoise_seed12345_eta0.0_zc1_stepXXXXX.png
#     samples_fixednoise_seed12345_eta0.0_zc2_stepXXXXX.png

import os
import time
import math
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
OUT_DIR = "outputs/runs/train_cond_256"
IMAGE_SIZE = 256

BATCH_SIZE = 8
LR = 2e-4
TOTAL_STEPS = 30000
SAVE_EVERY = 2000

NUM_WORKERS = 2
SEED = 12345

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
    # x in [0,1] -> [-1,1]
    return x01 * 2.0 - 1.0


def z_to_class(z_norm: torch.Tensor) -> torch.Tensor:
    """
    z_norm: (B,1) float in [0,1]
    returns: (B,) long in {0,1,2}
      0 ~ low depth (like z~0.1)
      1 ~ mid depth (like z~0.5)
      2 ~ high depth (like z~0.9)
    """
    z = z_norm.view(-1)
    # thresholds chosen to mimic low/mid/high sampling
    cls = torch.zeros_like(z, dtype=torch.long)
    cls[z >= 0.33] = 1
    cls[z >= 0.66] = 2
    return cls


# =====================
# DATASET
# =====================
class LungCTCondDataset(Dataset):
    """
    Expects .npy files named like:
      coronacases_XXX_z####.npy

    Assumes array values already in [0,1] (your fixed pipeline).
    """
    def __init__(self, data_dir: str, image_size: int = 256):
        self.data_dir = data_dir
        self.image_size = image_size

        files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        files.sort()
        if len(files) == 0:
            raise RuntimeError(f"No .npy files found in: {data_dir}")
        self.files = files

        # build zmax per case_id for correct z_norm
        # case_id from coronacases_007, z from z0088
        self.zmax = {}
        for fn in self.files:
            case_id, z_idx = self._parse_name(fn)
            self.zmax[case_id] = max(self.zmax.get(case_id, 0), z_idx)

        # precompute z_norm list
        self.z_norm_list = []
        for fn in self.files:
            case_id, z_idx = self._parse_name(fn)
            zmax = max(1, self.zmax[case_id])
            self.z_norm_list.append(float(z_idx) / float(zmax))

    @staticmethod
    def _parse_name(filename: str):
        # coronacases_007_z0088.npy
        base = filename.replace(".npy", "")
        parts = base.split("_")
        case_id = parts[1]  # "007"
        z_part = parts[2]   # "z0088"
        z_idx = int(z_part[1:])
        return case_id, z_idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        path = os.path.join(self.data_dir, fn)

        arr = np.load(path).astype(np.float32)  # expected [0,1]
        x = torch.from_numpy(arr)[None, ...]    # (1,H,W)

        # resize to 256x256
        x = x.unsqueeze(0)  # (1,1,H,W)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = x.squeeze(0)    # (1,256,256)

        # clamp safety
        x = torch.clamp(x, 0.0, 1.0)
        x = to_minus1_plus1(x)  # [-1,1]

        z_norm = torch.tensor([self.z_norm_list[idx]], dtype=torch.float32)  # (1,)

        return {
            "pixel_values": x,         # (1,256,256) in [-1,1]
            "z_norm": z_norm,          # (1,)
            "filename": fn
        }


# =====================
# FIXED-NOISE SAMPLING
# =====================
@torch.no_grad()
def sample_fixednoise(
    unet: UNet2DModel,
    ddim: DDIMScheduler,
    class_id: int,
    batch_size: int = 16,
    image_size: int = 256,
    seed: int = 12345,
    eta: float = 0.0
) -> torch.Tensor:
    """
    Deterministic DDIM when eta=0 (stable comparison across checkpoints).
    Returns tensor in [-1,1], shape (B,1,H,W)
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed + class_id)
    x = torch.randn((batch_size, 1, image_size, image_size), generator=gen, device=DEVICE)

    ddim.set_timesteps(50, device=DEVICE)  # 50 DDIM steps for speed
    for t in ddim.timesteps:
        model_out = unet(x, t, class_labels=torch.full((batch_size,), class_id, device=DEVICE, dtype=torch.long)).sample
        step_out = ddim.step(model_out, t, x, eta=eta)
        x = step_out.prev_sample

    x = torch.clamp(x, -1.0, 1.0)
    return x


# =====================
# MAIN TRAIN
# =====================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------------
    # DATA
    # ---------------------
    ds = LungCTCondDataset(DATA_DIR, image_size=IMAGE_SIZE)
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
        num_class_embeds=3,  # z-classes: 0,1,2
    ).to(DEVICE)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ---------------------
    # RESUME (optional)
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
        # move optimizer tensors to GPU
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(DEVICE)

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

        x0 = batch["pixel_values"].to(DEVICE)       # (B,1,256,256) [-1,1]
        z = batch["z_norm"].to(DEVICE).view(-1, 1)  # (B,1)
        labels = z_to_class(z)                      # (B,) in {0,1,2}

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
                for cls_id in [0, 1, 2]:
                    samples = sample_fixednoise(
                        unet=unet,
                        ddim=ddim_scheduler,
                        class_id=cls_id,
                        batch_size=16,
                        image_size=IMAGE_SIZE,
                        seed=SEED,
                        eta=0.0
                    )
                    out_png = os.path.join(
                        OUT_DIR,
                        f"samples_fixednoise_seed{SEED}_eta0.0_zc{cls_id}_step{step}.png"
                    )
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

