"""FloodVAE architecture and masked reconstruction/KL loss.

Train separate models for each area, hydraulic variable, and grid scale.
Inputs are log1p/min-max normalized coarse-grid maps. The decoder reconstructs
coarse maps, not fine-grid surrogate labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _format_mask(mask, dtype=torch.float32, device=None):
    if mask is None:
        return None

    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, dtype=dtype, device=device)
    else:
        mask = mask.to(dtype=dtype, device=device)

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)  # [1,1,H,W] if input is [1,H,W]
    elif mask.ndim != 4:
        raise ValueError("basin_mask must have shape [H,W], [1,H,W], or [1,1,H,W]")

    if mask.shape[:2] != (1, 1) or not torch.isfinite(mask).all():
        raise ValueError("basin_mask must have shape [1,1,H,W] and finite values")
    if not torch.all((mask == 0) | (mask == 1)) or not torch.any(mask):
        raise ValueError("basin_mask must be binary with at least one valid cell")
    return mask


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FloodVAE(nn.Module):
    """
    Input:  x [B, 1, H, W], H/W are multiples of 16
    Output: x_hat [B, 1, H, W], mu [B, z_dim], logvar [B, z_dim]
    """
    def __init__(
        self,
        in_channels=1,
        base_ch=32,
        z_dim=64,
        input_hw=(256, 256),
        basin_mask=None,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError("This network uses one hydraulic variable per model.")
        if len(input_hw) != 2 or any(v < 16 or v % 16 for v in input_hw):
            raise ValueError("input_hw dimensions must be positive multiples of 16 (at least 16).")
        if base_ch < 1 or z_dim < 1:
            raise ValueError("base_ch and z_dim must be positive.")
        self.z_dim = z_dim
        self.input_hw = input_hw

        self.enc = nn.Sequential(
            ConvBlock(in_channels, base_ch, stride=2),      # /2
            ConvBlock(base_ch, base_ch * 2, stride=2),      # /4
            ConvBlock(base_ch * 2, base_ch * 4, stride=2),  # /8
            ConvBlock(base_ch * 4, base_ch * 8, stride=2),  # /16
        )

        self.enc.eval()  # Shape inference must not update BatchNorm statistics.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw[0], input_hw[1])
            h = self.enc(dummy)
            self.enc_shape = h.shape[1:]
            flat_dim = h.numel()
        self.enc.train()

        self.fc_mu = nn.Linear(flat_dim, z_dim)
        self.fc_logvar = nn.Linear(flat_dim, z_dim)
        self.fc_dec = nn.Linear(z_dim, flat_dim)

        c, _, _ = self.enc_shape
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c, base_ch * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),

            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),

            nn.Conv2d(base_ch, 1, kernel_size=3, padding=1),
            nn.Softplus(),  # Nonnegative normalized depth or scalar speed.
        )

        formatted_mask = _format_mask(basin_mask)
        if formatted_mask is not None and tuple(formatted_mask.shape[-2:]) != tuple(input_hw):
            raise ValueError("basin_mask and input_hw must match.")
        self.register_buffer("basin_mask", formatted_mask)

    def set_basin_mask(self, basin_mask):
        formatted = _format_mask(
            basin_mask,
            dtype=self.fc_mu.weight.dtype,
            device=self.fc_mu.weight.device,
        )

        if formatted is not None and tuple(formatted.shape[-2:]) != tuple(self.input_hw):
            raise ValueError("basin_mask and input_hw must match.")
        self.basin_mask = formatted

    def encode(self, x):
        h = self.enc(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 5.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), *self.enc_shape)
        return self.dec(h)

    def apply_output_mask(self, x_hat):
        if self.basin_mask is None:
            return x_hat
        return x_hat * self.basin_mask

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        x_hat = self.apply_output_mask(x_hat)  # output-only constraint
        return x_hat, mu, logvar

    @torch.no_grad()
    def encode_mu(self, x):
        mu, _ = self.encode(x)
        return mu


def beta_vae_loss(
    x,
    x_hat,
    mu,
    logvar,
    beta=5e-4,
    basin_mask=None,
    eps=1e-8,
):
    """
    Simplified masked beta-VAE loss:
    - reconstruction only on valid basin pixels (if mask provided)
    - no wet/dry reweighting
    """
    if basin_mask is None:
        valid = torch.ones_like(x)
    else:
        valid = _format_mask(basin_mask, dtype=x.dtype, device=x.device)
        valid = valid.expand_as(x)

    recon_map = F.l1_loss(x_hat, x, reduction="none")
    recon = (recon_map * valid).sum() / valid.sum().clamp_min(eps)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon, kl

if __name__ == "__main__":
    # Small synthetic forward/backward check, unrelated to manuscript results.
    torch.manual_seed(42)
    vae = FloodVAE(base_ch=4, z_dim=8, input_hw=(32, 32))
    x = torch.rand(2, 1, 32, 32)
    x_hat, mu, logvar = vae(x)
    loss, recon, kl = beta_vae_loss(x, x_hat, mu, logvar)
    loss.backward()
    print("Output and latent shapes:", tuple(x_hat.shape), tuple(mu.shape))
    print("Loss, masked MAE, KL:", loss.item(), recon.item(), kl.item())
