import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class HighConv(MessagePassing):
    """High-frequency graph convolution with gating mechanism (memory-optimized)."""

    def __init__(self, in_channels: int, out_channels: int, aggr: str = "add"):
        super().__init__(aggr=aggr)
        # Keep the SAME parameterization as your original:
        self.gate = nn.Linear(2 * in_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        num_nodes = x.size(0)

        # Precompute norm once (faster than doing it inside message)
        deg = degree(row, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(~torch.isfinite(deg_inv_sqrt), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm, size=(num_nodes, num_nodes))

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        # Compute gate(concat([x_i, x_j])) WITHOUT torch.cat:
        # gate_weight: (1, 2F) → split into (1, F) + (1, F)
        W = self.gate.weight
        b = self.gate.bias

        F = x_i.size(-1)
        W_i = W[:, :F]      # (1, F)
        W_j = W[:, F:]      # (1, F)

        alpha_g = torch.tanh(x_i.matmul(W_i.t()) + x_j.matmul(W_j.t()) + b)  # (E, 1)

        # Reduce temporary allocations (compute scalar then scale x_j)
        scale = (norm.view(-1, 1) * alpha_g)          # (E, 1)
        return x_j * scale                            # (E, F)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out
