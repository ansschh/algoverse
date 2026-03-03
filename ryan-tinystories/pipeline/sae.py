"""
SparseAutoencoder: Anthropic "Towards Monosemanticity" design.

Architecture:
    x_centered = x - b_pre                       # (d_model,)
    h          = ReLU(W_enc @ x_centered + b_enc) # (n_features,) sparse hidden
    x_hat      = W_dec @ h + b_pre                # (d_model,)   reconstruction

    Loss = ||x - x_hat||² + λ * ||h||₁

Constraints from paper:
    - b_pre initialized to geometric median of training activations (not a grad param)
    - W_dec columns unit-normalized after every gradient step
    - Dead neuron resampling handled externally (see index_sae.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def geometric_median(X: torch.Tensor, n_iter: int = 50) -> torch.Tensor:
    """Weiszfeld algorithm for the geometric median of rows of X.

    Args:
        X: (n, d) tensor of data points
        n_iter: number of Weiszfeld iterations

    Returns:
        (d,) geometric median
    """
    if X.shape[0] == 1:
        return X[0].clone()

    y = X.mean(dim=0)
    for _ in range(n_iter):
        dists = torch.norm(X - y.unsqueeze(0), dim=1, keepdim=True).clamp(min=1e-8)
        weights = 1.0 / dists
        y = (weights * X).sum(dim=0) / weights.sum()
    return y


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with tied pre-encoder bias and unit-norm decoder columns.

    Args:
        d_model:    input / output dimension
        n_features: number of sparse hidden features (typically 2× d_model)
    """

    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        # Encoder: (n_features, d_model)
        self.W_enc = nn.Parameter(torch.empty(n_features, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: (d_model, n_features) — columns kept unit-norm
        self.W_dec = nn.Parameter(torch.empty(d_model, n_features))

        # Pre-encoder bias: NOT a gradient parameter; set by init_b_pre()
        self.register_buffer("b_pre", torch.zeros(d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        self.normalize_decoder()

    @torch.no_grad()
    def init_b_pre(self, X: torch.Tensor) -> None:
        """Set b_pre to the geometric median of X (n_samples, d_model)."""
        med = geometric_median(X.float())
        self.b_pre.copy_(med)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> h: (..., n_features) sparse activations."""
        return F.relu((x - self.b_pre) @ self.W_enc.T + self.b_enc)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., n_features) -> x_hat: (..., d_model) reconstruction."""
        return h @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        """Returns (h, x_hat)."""
        h = self.encode(x)
        x_hat = self.decode(h)
        return h, x_hat

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Normalize W_dec columns to unit norm in-place."""
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def loss(self, x: torch.Tensor, lambda_l1: float):
        """Compute SAE loss.

        Returns:
            total  : scalar loss (mse + lambda_l1 * l1)
            mse_val: float, reconstruction MSE
            l1_val : float, mean L1 of hidden activations
        """
        h, x_hat = self.forward(x)
        mse = F.mse_loss(x_hat, x)
        l1 = h.abs().mean()
        return mse + lambda_l1 * l1, mse.item(), l1.item()
