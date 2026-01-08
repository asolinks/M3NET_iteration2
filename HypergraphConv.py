from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)



class HypergraphConv(MessagePassing):
    """Hypergraph convolution with fixed indexing."""

    def __init__(self, in_channels: int, out_channels: int,
                 use_attention: bool = False, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, bias: bool = True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        # NOTE: edgeweight isn't used in your forward currently; keep for compatibility
        self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                EW_weight: Optional[Tensor] = None,
                dia_len: Optional[int] = None) -> Tensor:

        # Fast exits (no exceptions)
        if hyperedge_index.numel() == 0 or hyperedge_index.size(1) == 0:
            return x

        num_nodes = x.size(0)
        row, col = hyperedge_index[0], hyperedge_index[1]

        # If indices invalid, just return x (your original intent)
        if row.max().item() >= num_nodes:
            return x

        num_edges = int(col.max().item()) + 1

        if hyperedge_weight is None or hyperedge_weight.numel() == 0:
            hyperedge_weight = x.new_ones(num_edges)
        elif col.max().item() >= hyperedge_weight.size(0):
            # auto-fix sizing (same as yours, but no try/except)
            hyperedge_weight = x.new_ones(int(col.max().item()) + 1)

        alpha = None
        if self.use_attention:
            # NOTE: keeping your structure; but you likely want x projected by weight for correctness.
            # If changing this harms your trained behavior, keep as-is.
            assert hyperedge_attr is not None

            x_proj = torch.matmul(x, self.weight).view(-1, self.heads, self.out_channels)
            e_proj = torch.matmul(hyperedge_attr, self.weight).view(-1, self.heads, self.out_channels)

            x_i = x_proj[row]
            x_j = e_proj[col]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, row, num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

            x_in = x_proj
        else:
            x_in = x  # keep same behavior for non-attention

        # D: node-side normalization
        D = scatter_add(hyperedge_weight[col], row, dim=0, dim_size=num_nodes)
        D = D.reciprocal()
        D.masked_fill_(~torch.isfinite(D), 0.0)

        # B: edge-side normalization
        if EW_weight is None or EW_weight.numel() == 0 or EW_weight.size(0) != hyperedge_index.size(1):
            ew = x.new_ones(hyperedge_index.size(1))
        else:
            ew = EW_weight

        B = scatter_add(ew, col, dim=0, dim_size=num_edges)
        B = B.reciprocal()
        B.masked_fill_(~torch.isfinite(B), 0.0)

        # ---- Two-phase propagation WITHOUT cuda.empty_cache / exceptions ----
        # phase 1: nodes -> hyperedges (use B)
        self.flow = "source_to_target"
        out = self.propagate(hyperedge_index, x=x_in, norm=B, alpha=alpha, size=(num_nodes, num_edges))

        # phase 2: hyperedges -> nodes (use D)
        self.flow = "target_to_source"
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=None, size=(num_edges, num_nodes))

        # Output formatting (same intent)
        if self.concat and out.dim() == 3:
            out = out.reshape(-1, self.heads * self.out_channels)
        elif out.dim() == 3:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return F.leaky_relu(out)

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Optional[Tensor]) -> Tensor:
        H, Fout = self.heads, self.out_channels

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, Fout)
        else:
            out = norm_i.view(-1, 1, 1) * x_j

        if alpha is not None:
            out = alpha.view(-1, H, 1) * out
        return out
