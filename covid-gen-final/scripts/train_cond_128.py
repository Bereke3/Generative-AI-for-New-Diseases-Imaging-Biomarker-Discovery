import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from src.dataset import LungCTDataset
from src.ddpm_cond import ConditionalDDPM
from src.unet_cond import UNetCond


# =====================
# CONFIG
# =====================
DATA_DIR = "data/raw/slices"
OUT_DIR = "outputs/runs/train_cond_128"
IMAGE_SIZE = 128
BATCH_SIZE = 8
LR = 1e-4
TOTAL_STEPS = 20000
SAVE_EVERY = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # =====================
    # DATA
    # =====================
    dataset = LungCTDataset(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        conditional=True   # <-- z conditioning
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    dl_iter = iter(dataloader)

    # =====================
    # MODEL
    # =====================
    model = UNetCond(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        cond_dim=1           # z ∈ [0,1]
    ).to(DEVICE)

    diffusion = ConditionalDDPM(
        timesteps=1000,
        device=DEVICE
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    step = 0
    start_time = time.time()

    # =====================
    # TRAIN LOOP
    # =====================
    while step < TOTAL_STEPS:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        step += 1

        x0 = batch["pixel_values"].to(DEVICE)
        z = batch["z_norm"].to(DEVICE)    # shape (B,1)

        t = torch.randint(
            0, diffusion.timesteps,
            (x0.size(0),),
            device=DEVICE
        ).long()

        loss = diffusion.loss(
            model=model,
            x0=x0,
            t=t,
            class_labels=z
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =====================
        # LOG
        # =====================
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[{step}/{TOTAL_STEPS}] loss={loss.item():.6f} elapsed={elapsed:.1f}s")

        # =====================
        # SAVE
        # =====================
        if step % SAVE_EVERY == 0:
            with torch.no_grad():
                samples = diffusion.sample(
                    model=model,
                    batch_size=16,
                    image_size=IMAGE_SIZE,
                    z=torch.tensor([0.1, 0.5, 0.9], device=DEVICE)
                )

                save_image(
                    (samples + 1) / 2,
                    f"{OUT_DIR}/samples_step{step}.png",
                    nrow=4
                )

            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                },
                f"{OUT_DIR}/ckpt_step_{step}.pt"
            )


if __name__ == "__main__":
    main()

