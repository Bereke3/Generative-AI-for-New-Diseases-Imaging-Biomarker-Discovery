# scripts/generate_final_synth.py
#
# Final synthetic dataset generation (the thing you used to create:
#   outputs/final_synth/zbin3 (core) + outputs/final_synth/zbin2 (secondary)
# and metadata.csv)
#
# This script:
# 1) Loads a trained checkpoint from:
#       outputs/runs/diffusers_zbin_256/ckpt_stepXXXX.pt
#    (you choose which step; default uses 40000 if exists)
# 2) Uses EMA weights (stored in checkpoint) for sampling stability
# 3) Generates N images per zbin using DDIM sampling with a fixed seed stream
# 4) Saves PNGs + a metadata.csv describing each image (source run, ckpt, zbin, seed, eta, steps)
#
# IMPORTANT:
# - This script does NOT require src.diffusers_wrapper.
# - It is standalone: directly uses diffusers UNet2DModel + DDIMScheduler.
#
# Output:
#   outputs/final_synth/
#     zbin3/000000.png ...
#     zbin2/000000.png ...
#     metadata.csv
#
# Run:
#   source .venv/bin/activate
#   python -m scripts.generate_final_synth

import os
import csv
import glob
import random
import numpy as np
import torch
from torchvision.utils import save_image

from diffusers import UNet2DModel, DDIMScheduler
from diffusers.training_utils import EMAModel


# =====================
# CONFIG
# =====================
RUN_DIR = "outputs/runs/diffusers_zbin_256"
OUT_DIR = "outputs/final_synth"

IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which checkpoint to use:
# - if CKPT_PATH is None: pick the latest ckpt_step*.pt in RUN_DIR
CKPT_PATH = None

# Generation counts (your run used 800 + 800)
N_ZBIN3 = 800
N_ZBIN2 = 800

# Sampling settings
DDIM_STEPS = 50
ETA = 0.0  # 0.0 deterministic; keep stable for dataset creation

# Seeds: stable reproducibility
BASE_SEED = 12345

# Save progress every K images
LOG_EVERY = 160


# =====================
# UTILS
# =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_latest_ckpt(run_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_step*.pt")))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {run_dir}")
    # Sort by integer step
    ckpts = sorted(ckpts, key=lambda p: int(os.path.basename(p).split("step")[1].split(".pt")[0]))
    return ckpts[-1]


@torch.no_grad()
def ddim_sample_batch(
    unet: UNet2DModel,
    ema: EMAModel,
    ddim: DDIMScheduler,
    zbin: int,
    seeds: list,
    steps: int,
    eta: float
) -> torch.Tensor:
    """
    Generate a batch of images for a given zbin, with one seed per image.
    """
    b = len(seeds)

    # Apply EMA weights for sampling
    ema.store(unet.parameters())
    ema.copy_to(unet.parameters())

    # Create per-sample noise with per-sample seed
    noises = []
    for s in seeds:
        gen = torch.Generator(device=DEVICE).manual_seed(int(s))
        noises.append(torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), generator=gen, device=DEVICE))
    x = torch.cat(noises, dim=0)

    ddim.set_timesteps(steps, device=DEVICE)
    for t in ddim.timesteps:
        eps = unet(
            x, t,
            class_labels=torch.full((b,), zbin, device=DEVICE, dtype=torch.long)
        ).sample
        x = ddim.step(eps, t, x, eta=eta).prev_sample

    x = torch.clamp(x, -1.0, 1.0)

    # Restore original params
    ema.restore(unet.parameters())
    return x


def main():
    set_seed(BASE_SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    out_z3 = os.path.join(OUT_DIR, "zbin3")
    out_z2 = os.path.join(OUT_DIR, "zbin2")
    os.makedirs(out_z3, exist_ok=True)
    os.makedirs(out_z2, exist_ok=True)

    ckpt_path = CKPT_PATH or pick_latest_ckpt(RUN_DIR)

    # LOAD CKPT
    state = torch.load(ckpt_path, map_location="cpu")

    # Build model same as training
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

    unet.load_state_dict(state["unet"])
    unet.eval()

    # EMA
    ema = EMAModel(unet.parameters(), decay=0.999)
    ema.load_state_dict(state["ema"])
    ema.to(DEVICE)

    # Scheduler
    ddim = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True
    )

    # Prepare metadata file
    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relpath", "zbin", "seed", "ddim_steps", "eta", "ckpt_path", "run_dir"]
        )
        writer.writeheader()

        # ---- Generate zbin3 core ----
        print("Generating zbin3 (core) ...")
        idx = 0
        while idx < N_ZBIN3:
            batch = min(16, N_ZBIN3 - idx)
            seeds = [BASE_SEED + 100000 + idx + i for i in range(batch)]
            samples = ddim_sample_batch(unet, ema, ddim, zbin=3, seeds=seeds, steps=DDIM_STEPS, eta=ETA)

            for i in range(batch):
                fn = f"{idx+i:06d}.png"
                rel = f"zbin3/{fn}"
                save_path = os.path.join(OUT_DIR, rel)

                # save in [0,1]
                save_image((samples[i:i+1] + 1) / 2, save_path)

                writer.writerow({
                    "relpath": rel,
                    "zbin": 3,
                    "seed": int(seeds[i]),
                    "ddim_steps": DDIM_STEPS,
                    "eta": ETA,
                    "ckpt_path": ckpt_path,
                    "run_dir": RUN_DIR
                })

            idx += batch
            if idx % LOG_EVERY == 0 or idx == N_ZBIN3:
                print(f"[zbin3_core] generated {idx}/{N_ZBIN3}")

        # ---- Generate zbin2 secondary ----
        print("Generating zbin2 (secondary) ...")
        idx = 0
        while idx < N_ZBIN2:
            batch = min(16, N_ZBIN2 - idx)
            seeds = [BASE_SEED + 200000 + idx + i for i in range(batch)]
            samples = ddim_sample_batch(unet, ema, ddim, zbin=2, seeds=seeds, steps=DDIM_STEPS, eta=ETA)

            for i in range(batch):
                fn = f"{idx+i:06d}.png"
                rel = f"zbin2/{fn}"
                save_path = os.path.join(OUT_DIR, rel)

                save_image((samples[i:i+1] + 1) / 2, save_path)

                writer.writerow({
                    "relpath": rel,
                    "zbin": 2,
                    "seed": int(seeds[i]),
                    "ddim_steps": DDIM_STEPS,
                    "eta": ETA,
                    "ckpt_path": ckpt_path,
                    "run_dir": RUN_DIR
                })

            idx += batch
            if idx % LOG_EVERY == 0 or idx == N_ZBIN2:
                print(f"[zbin2_secondary] generated {idx}/{N_ZBIN2}")

    print("DONE. Saved to:", OUT_DIR)
    print("Metadata:", meta_path)


if __name__ == "__main__":
    main()

