# model.py (revised, optimized, drop-in)

from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

#from model_GCN import TextCNN  # keep your existing import surface
from model_hyper import HyperGCN


# -------------------------
# Losses (vectorized + correct)
# -------------------------

class FocalLoss(nn.Module):
    """
    Correct focal loss for multi-class classification.

    Supports logits shaped:
      - (T, B, C) or (N, C)
    and labels shaped:
      - (T, B) or (N,)
    """
    def __init__(self, gamma: float = 2.5, alpha: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:  # (T,B,C) -> (TB,C)
            T, B, C = logits.shape
            logits_ = logits.reshape(T * B, C)
            labels_ = labels.reshape(T * B)
        else:
            logits_ = logits
            labels_ = labels

        # standard CE per item
        ce = F.cross_entropy(logits_, labels_, reduction="none")
        # pt = exp(-ce)
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt).pow(self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MaskedNLLLoss(nn.Module):
    """NLL loss with masking support (kept API, faster + safer numerics)."""
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.loss = nn.NLLLoss(weight=weight, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # pred: (N,C), target: (N,), mask: (N,)
        mask = mask.reshape(-1).to(dtype=pred.dtype)
        nll = self.loss(pred, target)  # (N,)
        nll = nll * mask

        denom = mask.sum().clamp_min(1.0)
        if self.weight is not None:
            denom = (self.weight[target] * mask).sum().clamp_min(1.0)

        return nll.sum() / denom


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=pred.dtype)
        mse = self.loss(pred, target)  # same shape
        mse = mse * mask
        denom = mask.sum().clamp_min(1.0)
        return mse.sum() / denom


class UnMaskedWeightedNLLLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.loss = nn.NLLLoss(weight=weight, reduction="none")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nll = self.loss(pred, target)  # (N,)
        if self.weight is None:
            return nll.mean()
        denom = self.weight[target].sum().clamp_min(1.0)
        return nll.sum() / denom


# -------------------------
# Attention blocks (keep API, remove waste)
# -------------------------

class SimpleAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.scalar = nn.Linear(input_dim, 1, bias=False)

    def forward(self, M: torch.Tensor, x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # M: (T,B,D)
        scale = self.scalar(M)                 # (T,B,1)
        alpha = F.softmax(scale, dim=0)        # (T,B,1)
        alpha = alpha.permute(1, 2, 0)         # (B,1,T)
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # (B,D)
        return attn_pool, alpha


class MatchingAttention(nn.Module):
    def __init__(self, mem_dim: int, cand_dim: int, alpha_dim: Optional[int] = None, att_type: str = "general"):
        super().__init__()
        assert att_type != "concat" or alpha_dim is not None
        assert att_type != "dot" or mem_dim == cand_dim

        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type

        if att_type == "general":
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        elif att_type == "general2":
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == "concat":
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(
        self, M: torch.Tensor, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # M: (T,B,Dm), x: (B,Dc), mask: (B,T)
        T, B, Dm = M.shape
        if mask is None:
            mask = torch.ones(B, T, device=M.device, dtype=M.dtype)

        if self.att_type in ("dot", "general", "general2"):
            M_ = M.permute(1, 2, 0)  # (B,Dm,T)
            if self.att_type == "dot":
                x_ = x.unsqueeze(1)  # (B,1,Dm)
            else:
                x_ = self.transform(x).unsqueeze(1)  # (B,1,Dm)

            # raw scores: (B,1,T)
            scores = torch.bmm(x_, M_)  # (B,1,T)

            if self.att_type == "general2":
                # apply mask before softmax (avoid huge repeats)
                scores = scores * mask.unsqueeze(1)  # (B,1,T)
                scores = torch.tanh(scores)
                # masked softmax
                scores = scores.masked_fill(mask.unsqueeze(1) <= 0, float("-inf"))
                alpha = F.softmax(scores, dim=2)
                alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
            else:
                scores = scores.masked_fill(mask.unsqueeze(1) <= 0, float("-inf"))
                alpha = F.softmax(scores, dim=2)
                alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)

        else:
            # concat
            M_ = M.transpose(0, 1)  # (B,T,Dm)
            x_ = x.unsqueeze(1).expand(-1, T, -1)  # (B,T,Dc)
            M_x_ = torch.cat([M_, x_], dim=2)      # (B,T,Dm+Dc)
            mx_a = torch.tanh(self.transform(M_x_))  # (B,T,alpha_dim)
            alpha = F.softmax(self.vector_prod(mx_a), dim=1).transpose(1, 2)  # (B,1,T)

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # (B,Dm)
        return attn_pool, alpha


class Attention(nn.Module):
    """
    Kept for compatibility; fixes a major bug:
      - softmax must be over the key-length dimension (last dim), not dim=0.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        n_head: int = 1,
        score_function: str = "dot_product",
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function

        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        if score_function == "mlp":
            self.weight = nn.Parameter(torch.empty(hidden_dim * 2))
        elif score_function == "bi_linear":
            self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            stdv = 1.0 / math.sqrt(self.hidden_dim)
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k: torch.Tensor, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.dim() == 2:
            q = q.unsqueeze(1)
        if k.dim() == 2:
            k = k.unsqueeze(1)

        mb_size = k.size(0)
        k_len = k.size(1)
        q_len = q.size(1)

        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)

        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)

        if self.score_function in ("dot_product", "scaled_dot_product"):
            kt = kx.transpose(1, 2)  # (HB, Dh, K)
            score = torch.bmm(qx, kt)  # (HB, Q, K)
            if self.score_function == "scaled_dot_product":
                score = score / math.sqrt(self.hidden_dim)
        elif self.score_function == "mlp":
            kxx = kx.unsqueeze(1).expand(-1, q_len, -1, -1)
            qxx = qx.unsqueeze(2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)
            score = torch.tanh(torch.matmul(kq, self.weight))  # (HB,Q,K)
        elif self.score_function == "bi_linear":
            qw = torch.matmul(qx, self.weight)
            kt = kx.transpose(1, 2)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError("invalid score_function")

        # FIX: softmax over keys dimension
        score = F.softmax(score, dim=-1)

        output = torch.bmm(score, kx)  # (HB,Q,Dh)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (B,Q,H*Dh)
        output = self.proj(output)
        output = self.dropout(output)
        return output, score


# -------------------------
# Sequence models (kept API)
# -------------------------

class GRUModel(nn.Module):
    def __init__(self, D_m: int, D_e: int, D_h: int, n_classes: int = 7, dropout: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U: torch.Tensor, qmask: torch.Tensor, umask: torch.Tensor, att2: bool = True):
        emotions, _ = self.gru(U)  # (T,B,2De)

        if att2:
            # Keep semantics; still has a loop by design (it attends over full sequence).
            # You can remove it later with a batched attention kernel, but this version avoids extra waste.
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            alpha = []
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), dim=2)
        return log_prob, alpha, [], [], emotions


class LSTMModel(nn.Module):
    def __init__(self, D_m: int, D_e: int, D_h: int, n_classes: int = 7, dropout: float = 0.5):
        super().__init__()
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2 * D_e, 2 * D_e, att_type="general2")
        self.linear = nn.Linear(2 * D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U: torch.Tensor, qmask: torch.Tensor, umask: torch.Tensor, att2: bool = True):
        emotions, _ = self.lstm(U)

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            alpha = []
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), dim=2)
        return log_prob, alpha, [], [], emotions


# -------------------------
# Graphify helpers (FULLY VECTORIZED)
# -------------------------

def pad(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if length <= tensor.size(0):
        return tensor
    pad_shape = (length - tensor.size(0),) + tensor.size()[1:]
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)


def simple_batch_graphify(
    features: torch.Tensor, lengths: List[int], no_cuda: bool
) -> Tuple[torch.Tensor, None, None, None, None]:
    """
    Old: Python loop + cat.
    New: fully vectorized mask flatten.

    features: (T,B,D)
    lengths: list[int] len=B
    returns node_features: (sum(lengths), D)
    """
    # device already on features; ignore no_cuda and DO NOT call .cuda() here
    T, B, D = features.shape
    lengths_t = torch.as_tensor(lengths, device=features.device, dtype=torch.long)  # (B,)
    t_idx = torch.arange(T, device=features.device).unsqueeze(0)  # (1,T)
    mask = t_idx < lengths_t.unsqueeze(1)  # (B,T)

    # (B,T,D) then boolean-index => (sumL, D)
    node_features = features.permute(1, 0, 2)[mask]
    return node_features, None, None, None, None


# -------------------------
# Edge Attention (remove numpy + python overhead)
# -------------------------

class MaskedEdgeAttention(nn.Module):
    def __init__(self, input_dim: int, max_seq_len: int, no_cuda: bool):
        super().__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.scalar = nn.Linear(self.input_dim, self.max_seq_len, bias=False)
        self.no_cuda = no_cuda

    def forward(self, M: torch.Tensor, lengths: List[int], edge_ind: List[List[Tuple[int, int]]]) -> torch.Tensor:
        """
        Keeps your attn1 behavior, but:
          - no numpy
          - index build on torch
          - mask fill is GPU-friendly
        """
        # M: (T,B,D)
        scale = self.scalar(M)                        # (T,B,max_seq_len)
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # (B,max_seq_len,T)

        device = M.device
        B, S, T = alpha.shape

        mask = torch.full((B, S, T), 1e-10, device=device, dtype=alpha.dtype)
        mask_copy = torch.zeros((B, S, T), device=device, dtype=alpha.dtype)

        # build (3, E) indices: [batch, src, dst] to match your original meaning
        # original code: for each i (dialogue in batch), for each (src,dst) in edge_ind[i]
        b_idx = []
        s_idx = []
        t_idx = []
        for i, edges in enumerate(edge_ind):
            if not edges:
                continue
            e = torch.as_tensor(edges, device=device, dtype=torch.long)  # (Ei,2)
            b_idx.append(torch.full((e.size(0),), i, device=device, dtype=torch.long))
            s_idx.append(e[:, 0])
            t_idx.append(e[:, 1])

        if b_idx:
            b_idx = torch.cat(b_idx, dim=0)
            s_idx = torch.cat(s_idx, dim=0).clamp(0, S - 1)
            t_idx = torch.cat(t_idx, dim=0).clamp(0, T - 1)

            mask[b_idx, s_idx, t_idx] = 1.0
            mask_copy[b_idx, s_idx, t_idx] = 1.0

        masked_alpha = alpha * mask
        sums = masked_alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scores = (masked_alpha / sums) * mask_copy
        return scores


# -------------------------
# Multi-modal fusion (kept)
# -------------------------

class MMGatedAttention(nn.Module):
    def __init__(self, mem_dim: int, cand_dim: int, att_type: str = "general"):
        super().__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type

        self.dropouta = nn.Dropout(0.5)
        self.dropoutv = nn.Dropout(0.5)
        self.dropoutl = nn.Dropout(0.5)

        if att_type == "av_bg_fusion":
            self.transform_al = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_al = nn.Linear(mem_dim, cand_dim)
            self.transform_vl = nn.Linear(mem_dim * 2, cand_dim, bias=True)
            self.scalar_vl = nn.Linear(mem_dim, cand_dim)
        elif att_type == "general":
            self.transform_l = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_v = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_a = nn.Linear(mem_dim, cand_dim, bias=True)
            self.transform_av = nn.Linear(mem_dim * 3, 1)
            self.transform_al = nn.Linear(mem_dim * 3, 1)
            self.transform_vl = nn.Linear(mem_dim * 3, 1)
        else:
            raise ValueError(f"Unknown attention type: {att_type}")

    def forward(self, a: torch.Tensor, v: torch.Tensor, l: torch.Tensor, modals: Optional[List[str]] = None) -> torch.Tensor:
        if modals is None:
            modals = ["a", "v", "l"]

        a = self.dropouta(a) if isinstance(a, torch.Tensor) and a.numel() > 0 else a
        v = self.dropoutv(v) if isinstance(v, torch.Tensor) and v.numel() > 0 else v
        l = self.dropoutl(l) if isinstance(l, torch.Tensor) and l.numel() > 0 else l

        if self.att_type == "av_bg_fusion":
            return self._av_bg_fusion(a, v, l, modals)
        return self._general_fusion(a, v, l, modals)

    def _av_bg_fusion(self, a: torch.Tensor, v: torch.Tensor, l: torch.Tensor, modals: List[str]) -> torch.Tensor:
        hmf_components = [l]
        if "a" in modals:
            fal = torch.cat([a, l], dim=-1)
            Wa = torch.sigmoid(self.transform_al(fal))
            hma = Wa * self.scalar_al(a)
            hmf_components.append(hma)
        if "v" in modals:
            fvl = torch.cat([v, l], dim=-1)
            Wv = torch.sigmoid(self.transform_vl(fvl))
            hmv = Wv * self.scalar_vl(v)
            hmf_components.append(hmv)
        return torch.cat(hmf_components, dim=-1)

    def _general_fusion(self, a: torch.Tensor, v: torch.Tensor, l: torch.Tensor, modals: List[str]) -> torch.Tensor:
        ha = torch.tanh(self.transform_a(a)) if "a" in modals else a
        hv = torch.tanh(self.transform_v(v)) if "v" in modals else v
        hl = torch.tanh(self.transform_l(l)) if "l" in modals else l

        fusion_results = []

        if "a" in modals and "v" in modals:
            z_av = torch.sigmoid(self.transform_av(torch.cat([a, v, a * v], dim=-1)))
            h_av = z_av * ha + (1 - z_av) * hv
            if "l" not in modals:
                return h_av
            fusion_results.append(h_av)

        if "a" in modals and "l" in modals:
            z_al = torch.sigmoid(self.transform_al(torch.cat([a, l, a * l], dim=-1)))
            h_al = z_al * ha + (1 - z_al) * hl
            if "v" not in modals:
                return h_al
            fusion_results.append(h_al)

        if "v" in modals and "l" in modals:
            z_vl = torch.sigmoid(self.transform_vl(torch.cat([v, l, v * l], dim=-1)))
            h_vl = z_vl * hv + (1 - z_vl) * hl
            if "a" not in modals:
                return h_vl
            fusion_results.append(h_vl)

        return torch.cat(fusion_results, dim=-1)


# -------------------------
# Main Model (remove forward-time module creation, keep API)
# -------------------------

class Model(nn.Module):
    def __init__(
        self,
        base_model: str,
        D_m: int,
        D_g: int,
        D_p: int,
        D_e: int,
        D_h: int,
        D_a: int,
        graph_hidden_size: int,
        n_speakers: int,
        max_seq_len: int,
        window_past: int,
        window_future: int,
        n_classes: int = 7,
        listener_state: bool = False,
        context_attention: str = "simple",
        dropout_rec: float = 0.5,
        dropout: float = 0.5,
        nodal_attention: bool = True,
        avec: bool = False,
        no_cuda: bool = False,
        graph_type: str = "relation",
        use_topic: bool = False,
        alpha: float = 0.2,
        multiheads: int = 6,
        graph_construct: str = "direct",
        use_GCN: bool = False,
        use_residue: bool = True,
        dynamic_edge_w: bool = False,
        D_m_v: int = 512,
        D_m_a: int = 100,
        modals: str = "avl",
        att_type: str = "gated",
        av_using_lstm: bool = False,
        Deep_GCN_nlayers: int = 64,
        dataset: str = "IEMOCAP",
        use_speaker: bool = True,
        use_modal: bool = False,
        norm: str = "LN2",
        num_L: int = 3,
        num_K: int = 4,
    ):
        super().__init__()

        self.base_model = base_model
        self.avec = avec
        self.no_cuda = no_cuda
        self.graph_type = graph_type
        self.alpha = alpha
        self.multiheads = multiheads
        self.graph_construct = graph_construct
        self.use_topic = use_topic
        self.dropout = dropout
        self.use_GCN = use_GCN
        self.use_residue = use_residue
        self.dynamic_edge_w = dynamic_edge_w
        self.return_feature = True
        self.modals = [x for x in modals]
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.att_type = att_type
        self.norm_strategy = norm
        self.dataset = dataset
        self.D_m_v = D_m_v
        self.D_m_a = D_m_a

        # RoBERTa feature dim in your pipeline is typically 1024
        self.roberta_dim = 1024

        # Pre-create norms ONCE (big win vs creating LayerNorm inside forward)
        self.normBNa = nn.BatchNorm1d(self.roberta_dim, affine=True)
        self.normBNb = nn.BatchNorm1d(self.roberta_dim, affine=True)
        self.normBNc = nn.BatchNorm1d(self.roberta_dim, affine=True)
        self.normBNd = nn.BatchNorm1d(self.roberta_dim, affine=True)

        self.normLNa = nn.LayerNorm(self.roberta_dim, elementwise_affine=True)
        self.normLNb = nn.LayerNorm(self.roberta_dim, elementwise_affine=True)
        self.normLNc = nn.LayerNorm(self.roberta_dim, elementwise_affine=True)
        self.normLNd = nn.LayerNorm(self.roberta_dim, elementwise_affine=True)

        # LN2: keep name, but implement as per-token layer norm (fast + standard)
        self.normLN2 = nn.LayerNorm(self.roberta_dim, elementwise_affine=False)

        # multi-modal switch
        self.multi_modal = self.att_type in ["gated", "concat_subsequently", "concat_DHT"]
        self.av_using_lstm = av_using_lstm if self.multi_modal else False
        self.use_bert_seq = False

        # Base model setup
        if self.base_model == "LSTM":
            self._setup_lstm_base(D_m, D_g, dropout, D_e)
        elif self.base_model == "GRU":
            self._setup_gru_base(D_m, D_g, dropout)
        elif self.base_model == "Transformer":
            self._setup_transformer_base(D_m, D_g)
        elif self.base_model == "None":
            self.base_linear = nn.Linear(D_m, 2 * D_e)
        else:
            raise ValueError("Base model must be one of LSTM/GRU/Transformer/None")

        # Graph model setup
        if self.graph_type == "hyper":
            self.graph_model = HyperGCN(
                a_dim=D_g, v_dim=D_g, l_dim=D_g, n_dim=D_g, nlayers=64,
                nhidden=graph_hidden_size, nclass=n_classes, dropout=self.dropout,
                lamda=0.5, alpha=0.1, variant=True, return_feature=self.return_feature,
                use_residue=self.use_residue, n_speakers=n_speakers, modals=self.modals,
                use_speaker=self.use_speaker, use_modal=self.use_modal,
                num_L=num_L, num_K=num_K,
            )
        elif self.graph_type == "None":
            if not self.multi_modal:
                self.graph_net = nn.Linear(2 * D_e, n_classes)
            else:
                if "a" in self.modals:
                    self.graph_net_a = nn.Linear(2 * D_e, graph_hidden_size)
                if "v" in self.modals:
                    self.graph_net_v = nn.Linear(2 * D_e, graph_hidden_size)
                if "l" in self.modals:
                    self.graph_net_l = nn.Linear(2 * D_e, graph_hidden_size)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        # Output heads
        if self.multi_modal:
            self.dropout_ = nn.Dropout(self.dropout)
            self.hidfc = nn.Linear(graph_hidden_size, n_classes)

            if self.att_type == "concat_subsequently":
                in_dim = (D_g + graph_hidden_size) * len(self.modals) if self.use_residue else (graph_hidden_size * len(self.modals))
                self.smax_fc = nn.Linear(in_dim, n_classes)
            elif self.att_type == "concat_DHT":
                in_dim = (D_g + graph_hidden_size * 2) * len(self.modals) if self.use_residue else ((graph_hidden_size * 2) * len(self.modals))
                self.smax_fc = nn.Linear(in_dim, n_classes)
            elif self.att_type == "gated":
                self.smax_fc = nn.Linear(100 * len(self.modals) if len(self.modals) == 3 else 100, graph_hidden_size)
            else:
                self.smax_fc = nn.Linear(D_g + graph_hidden_size * len(self.modals), graph_hidden_size)

    def _setup_lstm_base(self, D_m: int, D_g: int, dropout: float, D_e: int):
        if not self.multi_modal:
            hidden_ = 250 if len(self.modals) == 3 else 100
            if "".join(self.modals) in ("al", "vl"):
                hidden_ = 150
            self.linear_ = nn.Linear(D_m, hidden_)
            self.lstm = nn.LSTM(input_size=hidden_, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        else:
            if "a" in self.modals:
                self.linear_a = nn.Linear(self.D_m_a, D_g)
                if self.av_using_lstm:
                    self.lstm_a = nn.LSTM(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True, dropout=dropout)
            if "v" in self.modals:
                self.linear_v = nn.Linear(self.D_m_v, D_g)
                if self.av_using_lstm:
                    self.lstm_v = nn.LSTM(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True, dropout=dropout)
            if "l" in self.modals:
                if self.use_bert_seq:
                    self.txtCNN = TextCNN(input_dim=D_m, emb_size=D_g)
                else:
                    self.linear_l = nn.Linear(D_m, D_g)
                self.lstm_l = nn.LSTM(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True, dropout=dropout)

    def _setup_gru_base(self, D_m: int, D_g: int, dropout: float):
        if "a" in self.modals:
            self.linear_a = nn.Linear(self.D_m_a, D_g)
            if self.av_using_lstm:
                self.gru_a = nn.GRU(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True, dropout=dropout)
        if "v" in self.modals:
            self.linear_v = nn.Linear(self.D_m_v, D_g)
            if self.av_using_lstm:
                self.gru_v = nn.GRU(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True, dropout=dropout)
        if "l" in self.modals:
            if self.use_bert_seq:
                self.txtCNN = TextCNN(input_dim=D_m, emb_size=D_g)
            else:
                if self.dataset != "MELD":
                    self.linear_l = nn.Linear(D_m, D_g)
            self.gru_l = nn.GRU(input_size=D_g, hidden_size=D_g // 2, num_layers=2, bidirectional=True, dropout=dropout)

    def _setup_transformer_base(self, D_m: int, D_g: int):
        if "a" in self.modals:
            self.linear_a = nn.Linear(self.D_m_a, D_g)
            if self.av_using_lstm:
                self.trans_a = nn.TransformerEncoderLayer(d_model=D_g, nhead=4)
        if "v" in self.modals:
            self.linear_v = nn.Linear(self.D_m_v, D_g)
            if self.av_using_lstm:
                self.trans_v = nn.TransformerEncoderLayer(d_model=D_g, nhead=4)
        if "l" in self.modals:
            if self.use_bert_seq:
                self.txtCNN = TextCNN(input_dim=D_m, emb_size=D_g)
            else:
                self.linear_l = nn.Linear(D_m, D_g)
            self.trans_l = nn.TransformerEncoderLayer(d_model=D_g, nhead=4)

    def forward(self, U, qmask, umask, seq_lengths, U_a=None, U_v=None, epoch=None):
        # U is (r1,r2,r3,r4), each (T,B,1024)
        r1, r2, r3, r4 = U
        T, B, D = r1.shape

        # Fast norm helpers: flatten once, norm, reshape back
        def _bn(bn: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
            xb = x.transpose(0, 1).reshape(-1, D)     # (B*T, D)
            xb = bn(xb)
            return xb.view(B, T, D).transpose(0, 1)   # (T,B,D)

        def _ln(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
            return ln(x)  # LN over last dim is native on (T,B,D)

        if self.norm_strategy == "LN":
            r1 = _ln(self.normLNa, r1)
            r2 = _ln(self.normLNb, r2)
            r3 = _ln(self.normLNc, r3)
            r4 = _ln(self.normLNd, r4)
        elif self.norm_strategy == "BN":
            r1 = _bn(self.normBNa, r1)
            r2 = _bn(self.normBNb, r2)
            r3 = _bn(self.normBNc, r3)
            r4 = _bn(self.normBNd, r4)
        elif self.norm_strategy == "LN2":
            # old LN2 created a LayerNorm inside forward (slow).
            # Here: per-token LN (fast + stable)
            r1 = self.normLN2(r1)
            r2 = self.normLN2(r2)
            r3 = self.normLN2(r3)
            r4 = self.normLN2(r4)

        U_avg = (r1 + r2 + r3 + r4) * 0.25  # (T,B,D)

        # Base model forward
        if self.base_model == "LSTM":
            emotions = self._forward_lstm(U_avg, U_a, U_v)
        elif self.base_model == "GRU":
            emotions = self._forward_gru(U_avg, U_a, U_v)
        elif self.base_model == "Transformer":
            emotions = self._forward_transformer(U_avg, U_a, U_v)
        elif self.base_model == "None":
            emotions = self.base_linear(U_avg)
        else:
            raise ValueError(self.base_model)

        # Graph head
        if self.graph_type == "hyper":
            log_prob = self._forward_hyper_graph(emotions, seq_lengths, qmask, epoch)
        elif self.graph_type == "None":
            log_prob = self._forward_simple_graph(emotions, seq_lengths, qmask)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

        return log_prob, None, None, None, None

    def _forward_lstm(self, U, U_a, U_v):
        if not self.multi_modal:
            U = self.linear_(U)
            emotions, _ = self.lstm(U)
            return emotions
        return self._forward_multi_modal(U, U_a, U_v, kind="lstm")

    def _forward_gru(self, U, U_a, U_v):
        return self._forward_multi_modal(U, U_a, U_v, kind="gru")

    def _forward_transformer(self, U, U_a, U_v):
        return self._forward_multi_modal(U, U_a, U_v, kind="trans")

    def _forward_multi_modal(self, U, U_a, U_v, kind: str) -> Dict[str, torch.Tensor]:
        emotions_dict: Dict[str, torch.Tensor] = {}

        if "a" in self.modals:
            xa = self.linear_a(U_a)
            if self.av_using_lstm:
                if kind == "lstm":
                    xa, _ = self.lstm_a(xa)
                elif kind == "gru":
                    xa, _ = self.gru_a(xa)
                else:
                    xa = self.trans_a(xa)
            emotions_dict["a"] = xa

        if "v" in self.modals:
            xv = self.linear_v(U_v)
            if self.av_using_lstm:
                if kind == "lstm":
                    xv, _ = self.lstm_v(xv)
                elif kind == "gru":
                    xv, _ = self.gru_v(xv)
                else:
                    xv = self.trans_v(xv)
            emotions_dict["v"] = xv

        if "l" in self.modals:
            if self.use_bert_seq:
                U_ = U.reshape(-1, U.size(-2), U.size(-1))
                xl = self.txtCNN(U_).reshape(U.size(0), U.size(1), -1)
            else:
                if self.base_model == "GRU" and self.dataset == "MELD":
                    xl = U
                else:
                    xl = self.linear_l(U)

            if kind == "lstm":
                xl, _ = self.lstm_l(xl)
            elif kind == "gru":
                xl, _ = self.gru_l(xl)
            else:
                xl = self.trans_l(xl)
            emotions_dict["l"] = xl

        return emotions_dict

    def _forward_hyper_graph(self, emotions, seq_lengths, qmask, epoch):
        if not self.multi_modal:
            features, _, _, _, _ = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
            emotions_feat = self.graph_model(features, features, features, seq_lengths, qmask, epoch)
        else:
            fa, fv, fl = self._prepare_multi_modal_features(emotions, seq_lengths)
            emotions_feat = self.graph_model(fa, fv, fl, seq_lengths, qmask, epoch)

        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = F.relu(emotions_feat)
        return F.log_softmax(self.smax_fc(emotions_feat), dim=1)

    def _forward_simple_graph(self, emotions, seq_lengths, qmask):
        if not self.multi_modal:
            features, _, _, _, _ = simple_batch_graphify(emotions, seq_lengths, self.no_cuda)
            # old code expected a graph net; keep simple linear behavior
            return F.log_softmax(self.graph_net(features), dim=1)

        fa, fv, fl = self._prepare_multi_modal_features(emotions, seq_lengths)
        out_list = []
        if "a" in self.modals:
            out_list.append(self.graph_net_a(fa))
        if "v" in self.modals:
            out_list.append(self.graph_net_v(fv))
        if "l" in self.modals:
            out_list.append(self.graph_net_l(fl))

        emotions_feat = torch.cat(out_list, dim=-1)
        emotions_feat = self.dropout_(emotions_feat)
        emotions_feat = F.relu(emotions_feat)
        return F.log_softmax(self.hidfc(self.smax_fc(emotions_feat)), dim=1)

    def _prepare_multi_modal_features(self, emotions: Dict[str, torch.Tensor], seq_lengths: List[int]):
        fa = fv = fl = []
        if "a" in self.modals:
            fa, _, _, _, _ = simple_batch_graphify(emotions["a"], seq_lengths, self.no_cuda)
        if "v" in self.modals:
            fv, _, _, _, _ = simple_batch_graphify(emotions["v"], seq_lengths, self.no_cuda)
        if "l" in self.modals:
            fl, _, _, _, _ = simple_batch_graphify(emotions["l"], seq_lengths, self.no_cuda)
        return fa, fv, fl


__all__ = [
    "FocalLoss",
    "MaskedNLLLoss",
    "MaskedMSELoss",
    "UnMaskedWeightedNLLLoss",
    "SimpleAttention",
    "MatchingAttention",
    "Attention",
    "GRUModel",
    "LSTMModel",
    "MaskedEdgeAttention",
    "MMGatedAttention",
    "Model",
]
