
# scripts/train_diffusers_zbin_cfg.py
#
# Classifier-Free Guidance (CFG) training on top of the baseline z-bin model.
#
# Key idea:
# - Baseline model: num_class_embeds=4 (zbin 0..3)
# - CFG model: num_class_embeds=5 where class_id=4 is the "null/unconditional" class
# - During training, with probability p_uncond we drop the condition:
#       labels_used = NULL_CLASS (4)
#   else:
#       labels_used = real zbin label (0..3)
#
# Loading baseline -> cfg:
# - copy all weights strictly except class_embedding.weight
# - for class_embedding.weight:
#     cfg_w[:4] = baseline_w
#     cfg_w[4]  = mean(baseline_w)   (stable initialization for null class)
#
# Outputs:
#   outputs/runs/diffusers_zbin_256_cfg/
#     ckpt_stepXXXX.pt
#     samples_cfg_w{w}_zbin{zb}_stepXXXX.png   (for several w values)
#
# Resume:
# - resumes from latest checkpoint in run_dir
# - handles case where checkpoint has no "ema" (safe)
#
# NOTE:
# - This file is the "fixed" version that will NOT crash at ema.load_state_dict when ema is None.

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
BASELINE_CKPT = "outputs/runs/diffusers_zbin_256/ckpt_step40000.pt"  # change if needed

RUN_DIR = "outputs/runs/diffusers_zbin_256_cfg"

IMAGE_SIZE = 256
BATCH_SIZE = 8
LR = 2e-4
TOTAL_STEPS = 20000
SAVE_EVERY = 2000
NUM_WORKERS = 2

SEED = 12345
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CFG
NULL_CLASS = 4          # 5th embedding index
P_UNCOND = 0.10         # 10% unconditional drop during training
CFG_WEIGHTS = [1.0, 2.0, 3.0]  # sampling weights to export

# DDIM sampling
SAMPLE_STEPS = 50
SAMPLE_ETA = 0.0        # deterministic for CFG evaluation


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


def move_optimizer_to_device(opt: torch.optim.Optimizer, device: str):
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


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
            "pixel_values": x,
            "class_labels": torch.tensor(zbin).long()
        }


# =====================
# BASELINE -> CFG INIT
# =====================
def load_baseline_with_null_class(unet_cfg: UNet2DModel, baseline_unet_state: dict):
    """
    baseline_unet_state: state_dict from baseline with num_class_embeds=4
    unet_cfg: model with num_class_embeds=5
    """
    # load everything except class_embedding.weight
    filtered = {k: v for k, v in baseline_unet_state.items() if k != "class_embedding.weight"}
    unet_cfg.load_state_dict(filtered, strict=False)

    if "class_embedding.weight" not in baseline_unet_state:
        raise RuntimeError("Baseline checkpoint missing class_embedding.weight")

    w4 = baseline_unet_state["class_embedding.weight"]  # [4, D]
    if w4.dim() != 2 or w4.shape[0] != 4:
        raise RuntimeError(f"Unexpected baseline class_embedding.weight shape: {tuple(w4.shape)}")

    # Build [5, D]
    with torch.no_grad():
        w5 = unet_cfg.class_embedding.weight.data  # [5, D]
        if w5.shape[0] != 5 or w5.shape[1] != w4.shape[1]:
            raise RuntimeError(f"CFG class_embedding shape mismatch: {tuple(w5.shape)} vs baseline {tuple(w4.shape)}")

        w5[:4].copy_(w4)
        w5[4].copy_(w4.mean(dim=0))  # stable null init


# =====================
# SAMPLING (CFG)
# =====================
@torch.no_grad()
def sample_cfg(
    unet: UNet2DModel,
    ema: EMAModel,
    ddim: DDIMScheduler,
    zbin: int,
    cfg_w: float,
    batch_size: int = 16,
    seed: int = 12345,
    steps: int = 50,
    eta: float = 0.0
) -> torch.Tensor:
    """
    CFG sampling:
        eps = eps_uncond + w * (eps_cond - eps_uncond)
    where uncond uses class label = NULL_CLASS (4)
    """
    ema.store(unet.parameters())
    ema.copy_to(unet.parameters())

    gen = torch.Generator(device=DEVICE).manual_seed(seed + 10 * zbin + int(cfg_w * 100))
    x = torch.randn((batch_size, 1, IMAGE_SIZE, IMAGE_SIZE), generator=gen, device=DEVICE)

    ddim.set_timesteps(steps, device=DEVICE)
    for t in ddim.timesteps:
        t_in = t

        eps_uncond = unet(
            x, t_in,
            class_labels=torch.full((batch_size,), NULL_CLASS, device=DEVICE, dtype=torch.long)
        ).sample

        eps_cond = unet(
            x, t_in,
            class_labels=torch.full((batch_size,), zbin, device=DEVICE, dtype=torch.long)
        ).sample

        eps = eps_uncond + cfg_w * (eps_cond - eps_uncond)
        x = ddim.step(eps, t_in, x, eta=eta).prev_sample

    x = torch.clamp(x, -1.0, 1.0)

    ema.restore(unet.parameters())
    return x


# =====================
# CKPT IO
# =====================
def save_ckpt(path: str, step: int, unet: UNet2DModel, opt: torch.optim.Optimizer, ema: EMAModel):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "unet": unet.state_dict(),
            "opt": opt.state_dict(),
            "ema": (ema.state_dict() if ema is not None else None),
        },
        path
    )


def load_ckpt(path: str, unet: UNet2DModel, opt: torch.optim.Optimizer, ema: EMAModel):
    state = torch.load(path, map_location="cpu")
    step = int(state.get("step", 0))

    unet.load_state_dict(state["unet"])
    opt.load_state_dict(state["opt"])

    # SAFE: checkpoint might store ema=None
    ema_state = state.get("ema", None)
    if ema is not None and ema_state is not None:
        ema.load_state_dict(ema_state)

    return step


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

    # SCHEDULERS
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    ddim_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", clip_sample=True)

    # MODEL (CFG = 5 classes)
    unet = UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        num_class_embeds=5,  # 0..3 bins + 4 null
    ).to(DEVICE)

    opt = torch.optim.AdamW(unet.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    ema = EMAModel(unet.parameters(), decay=0.999)
    ema.to(DEVICE)

    # RESUME
    step = 0
    ckpts = [f for f in os.listdir(RUN_DIR) if f.startswith("ckpt_step") and f.endswith(".pt")]
    if ckpts:
        ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("step")[1].split(".pt")[0]))
        last_ckpt = os.path.join(RUN_DIR, ckpts_sorted[-1])
        print("Resuming CFG from:", last_ckpt)
        step = load_ckpt(last_ckpt, unet, opt, ema)
        move_optimizer_to_device(opt, DEVICE)
        ema.to(DEVICE)
    else:
        print("No CFG checkpoint found. Initializing from baseline:", BASELINE_CKPT)
        if not os.path.exists(BASELINE_CKPT):
            raise RuntimeError(f"Baseline checkpoint not found: {BASELINE_CKPT}")
        base = torch.load(BASELINE_CKPT, map_location="cpu")
        baseline_state = base["unet"] if "unet" in base else base
        load_baseline_with_null_class(unet, baseline_state)

    t0 = time.time()

    # TRAIN
    while step < TOTAL_STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        step += 1
        x0 = batch["pixel_values"].to(DEVICE)
        labels = batch["class_labels"].to(DEVICE).view(-1)  # 0..3

        # Drop condition for CFG training
        drop_mask = (torch.rand(labels.shape[0], device=DEVICE) < P_UNCOND)
        labels_used = labels.clone()
        labels_used[drop_mask] = NULL_CLASS

        noise = torch.randn_like(x0)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x0.size(0),), device=DEVICE).long()
        x_t = noise_scheduler.add_noise(x0, noise, t)

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = unet(x_t, t, class_labels=labels_used).sample
            loss = torch.mean((pred - noise) ** 2)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        ema.step(unet.parameters())

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"[{step}/{TOTAL_STEPS}] loss={loss.item():.6f} elapsed={elapsed:.1f}s")

        if step % SAVE_EVERY == 0:
            unet.eval()
            with torch.no_grad():
                for zb in range(4):
                    for w in CFG_WEIGHTS:
                        samples = sample_cfg(
                            unet=unet,
                            ema=ema,
                            ddim=ddim_scheduler,
                            zbin=zb,
                            cfg_w=w,
                            batch_size=16,
                            seed=SEED,
                            steps=SAMPLE_STEPS,
                            eta=SAMPLE_ETA
                        )
                        out_png = os.path.join(RUN_DIR, f"samples_cfg_w{w}_zbin{zb}_step{step}.png")
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
