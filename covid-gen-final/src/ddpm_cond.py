import torch
import torch.nn.functional as F

class ConditionalDDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.device = device
        self.timesteps = timesteps

        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def loss(self, model, x0, t, class_labels):
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)

        pred_noise = model(x_noisy, t, class_labels=class_labels)
        return F.mse_loss(pred_noise, noise)

