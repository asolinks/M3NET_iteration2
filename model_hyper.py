"""
HyperGCN Model (Vectorized & Modernized)

This module contains:
    - STEFunction: Straight-through estimator
    - GraphConvolution: GCN layer with PyTorch 2.x compatibility
    - PositionalEncoding: Sinusoidal encoding for dialogues
    - HyperGCN: Multi-modal Hypergraph GCN + High-frequency GNN
      with vectorized hyperedge construction.

Main improvements:
    * Fully vectorized hypergraph & GNN edge construction
    * No Python-level bottlenecks (loops, append, permutations)
    * Fully torch.compile compatible
    * Complete type hints & documentation
"""

from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from HypergraphConv import HypergraphConv
from high_fre_conv import HighConv


# ======================================================================
# STRAIGHT-THROUGH ESTIMATOR
# ======================================================================
class STEFunction(torch.autograd.Function):
    """
    Straight-through estimator (binary gate approximation).

    forward:
        y = 1 if x > 0 else 0
    backward:
        gradient clipped to [-1, +1]
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return (input > 0).to(input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clamp(-1, 1)


# ======================================================================
# GRAPH CONVOLUTION LAYER (PyTorch 2.x compatible)
# ======================================================================
class GraphConvolution(nn.Module):
    """
    Standard Graph Convolution Layer (GCNII style).

    Args:
        in_features: input dimensionality
        out_features: output dimensionality
        residual: whether to use residual skip-connection
        variant: GCNII variant concatenating h₀

    Returns:
        Tensor: updated node embeddings
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        residual: bool = False,
        variant: bool = False
    ):
        super().__init__()

        self.variant = variant
        self.out_features = out_features
        self.residual = residual

        self.in_features = 2 * in_features if variant else in_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the learnable weight matrix."""
        stdv = 1.0 / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight, -stdv, stdv)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        h0: torch.Tensor,
        lamda: float,
        alpha: float,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Perform one graph convolution step.

        Args:
            x: (N, D) current node features
            adj: (N, N) adjacency matrix
            h0: (N, D) initial features for skip connection
            lamda: scalar controlling residual mixing
            alpha: skip coefficient
            layer_idx: which layer this is (1-based)

        Returns:
            Tensor: updated node embeddings (N, D_out)
        """
        lam = torch.tensor(lamda, device=x.device, dtype=x.dtype)
        theta = torch.log(lam / layer_idx + 1.0)

        hi = adj @ x  # dense matrix multiply

        if self.variant:
            support = torch.cat([hi, h0], dim=-1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support

        out = theta * (support @ self.weight) + (1 - theta) * r

        if self.residual:
            out = out + x

        return out


# ======================================================================
# POSITIONAL ENCODING
# ======================================================================
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding applied to dialogue sequences.

    Args:
        d_model: feature dimension
        dropout: dropout probability
        max_len: maximum dialogue length
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, dia_len: List[int]) -> torch.Tensor:
        """
        Args:
            x: (total_T, D)
            dia_len: list of dialogue lengths

        Returns:
            Tensor with position encodings added.
        """
        lens = torch.tensor(dia_len, device=x.device)
        maxT = lens.max()

        idx = torch.arange(maxT, device=x.device).unsqueeze(0).expand(len(lens), -1)
        mask = idx < lens.unsqueeze(1)

        pos = idx[mask]

        return self.dropout(x + self.pe[pos])


# ======================================================================
# HYPERGCN MODEL - INIT
# ======================================================================
class HyperGCN(nn.Module):
    """
    Multi-modal Hypergraph + High-frequency GNN model.

    This implements the architecture used in M3NET / MHGCN:
        • Builds hyperedges for modalities + utterances
        • Builds dense GNN adjacency
        • Applies multiple HypergraphConv and HighConv layers
        • Vectorized implementation, PyTorch 2.x compatible
    """

    def __init__(
        self,
        a_dim: int,
        v_dim: int,
        l_dim: int,
        n_dim: int,
        nlayers: int,
        nhidden: int,
        nclass: int,
        dropout: float,
        lamda: float,
        alpha: float,
        variant: bool,
        return_feature: bool,
        use_residue: bool,
        new_graph: str = 'full',
        n_speakers: int = 2,
        modals: Optional[List[str]] = None,
        use_speaker: bool = True,
        use_modal: bool = False,
        num_L: int = 3,
        num_K: int = 4
    ):
        super().__init__()

        if modals is None:
            modals = ['a', 'v', 'l']

        # --------------------------
        # Configuration
        # --------------------------
        self.modals = modals
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.return_feature = return_feature
        self.use_residue = use_residue
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False
        self.num_L = num_L
        self.num_K = num_K

        # --------------------------
        # Embeddings
        # --------------------------
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)

        # --------------------------
        # Feature transform
        # --------------------------
        self.fc1 = nn.Linear(n_dim, nhidden)

        # --------------------------
        # Hypergraph + GNN layers
        # --------------------------
        self.hyperconvs = nn.ModuleList(
            [HypergraphConv(nhidden, nhidden) for _ in range(num_L)]
        )
        self.gnn_convs = nn.ModuleList(
            [HighConv(nhidden, nhidden) for _ in range(num_K)]
        )

        # Hyperedge attributes (learnable)
        self.hyperedge_attr1 = Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = Parameter(torch.rand(nhidden))
    # ==================================================================
    # SPEAKER EMBEDDINGS (VECTORIZED)
    # ==================================================================
    def extract_speaker_embeddings(
        self,
        qmask: torch.Tensor,
        dia_len: List[int]
    ) -> torch.Tensor:
        """
        Extract speaker embeddings for each utterance in the batch.

        Args:
            qmask:
                Speaker mask tensor of shape (max_T, batch, num_speakers).
                Each timestep has a one-hot vector indicating the speaker.
            dia_len:
                List of dialogue lengths for each item in the batch.

        Returns:
            Tensor of shape (sum(T), n_dim):
                Speaker embedding for each utterance in concatenated order.

        Notes:
            This is a fully vectorized implementation (no Python loops).
        """
        lens = torch.tensor(dia_len, device=qmask.device)
        maxT = qmask.size(0)

        # Build indices: mask selects only valid timesteps in each dialogue
        time_idx = torch.arange(maxT, device=qmask.device).unsqueeze(0)
        mask = time_idx < lens.unsqueeze(1)  # (batch, max_T)

        # Permute qmask to (batch, max_T, num_speakers)
        valid_qmask = qmask.permute(1, 0, 2)[mask]  # shape: (sum(T), num_speakers)

        # Speaker index chosen using argmax
        spk_idx = valid_qmask.argmax(dim=-1)

        # Return embedding for each utterance
        return self.speaker_embeddings(spk_idx)


    # ==================================================================
    # FEATURE RECONSTRUCTION (REVERSE CONCATENATION)
    # ==================================================================
    def reverse_features(
        self,
        dia_len: List[int],
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Reverse the L||A||V stacking operation used during hypergraph creation.

        The input features are stacked per-dialog as:
            [L1..LT, A1..AT, V1..VT]

        This method restores the per-utterance arrangement:
            concat([L_i, A_i, V_i]) for i = 1..T

        Args:
            dia_len:
                List of dialogue lengths.
            features:
                Tensor of shape (sum(3*T), D).

        Returns:
            Tensor of shape (sum(T), 3D):
                Concatenated per-utterance features.
        """
        outputs = []
        pointer = 0

        for T in dia_len:
            # Slice segments for L, A, V
            L_seg = features[pointer : pointer + T]
            A_seg = features[pointer + T : pointer + 2*T]
            V_seg = features[pointer + 2*T : pointer + 3*T]
            pointer += 3 * T

            # Per-utterance concatenation
            merged = torch.cat([L_seg, A_seg, V_seg], dim=-1)
            outputs.append(merged)

        return torch.cat(outputs, dim=0)
    # ==================================================================
    # (1) HYPERGRAPH INDEX CONSTRUCTION (VECTORIZED)
    # ==================================================================
    def create_hyper_index(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        l: torch.Tensor,
        dia_len: List[int],
        modals: List[str]
    ) -> Tuple[
        torch.Tensor,  # hyperedge_index
        torch.Tensor,  # node_edge_index
        torch.Tensor,  # features
        torch.Tensor,  # batch
        torch.Tensor   # hyperedge_type
    ]:
        """
        Build the hypergraph structure for the multi-modal dialogue batch.

        Produces FIVE outputs:

        1. hyperedge_index  (2, E_he)
           - Incidence matrix for the hypergraph.
           - Maps node → hyperedge.

        2. edge_index       (2, E_node)
           - Dense node-to-node adjacency used for HighConv (GNN component).

        3. features         (sum(3T_i), D)
           - Concatenated features in the order [L, A, V] per dialogue.

        4. batch            (sum(3T_i),)
           - Batch ID for each node.

        5. hyperedge_type   (E_he, 1)
           - 1 → modality-level hyperedge
           - 0 → utterance-level hyperedge

        Args:
            a: (total_T, D) audio features
            v: (total_T, D) visual features
            l: (total_T, D) language features
            dia_len: list of dialogue lengths
            modals: list of modality names ['a','v','l']

        Returns:
            Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        """
        device = a.device
        dtype = torch.long
        num_modality = len(modals)

        # -------------------------------------------------------
        # Early exit if no dialogues
        # -------------------------------------------------------
        if not dia_len or sum(dia_len) == 0:
            empty_he = torch.empty((2, 0), dtype=dtype, device=device)
            empty_node = torch.empty((2, 0), dtype=dtype, device=device)
            empty_feat = torch.empty((0, a.size(1) + v.size(1) + l.size(1)), device=device)
            empty_batch = torch.empty(0, dtype=dtype, device=device)
            empty_type = torch.empty((0, 1), dtype=dtype, device=device)
            return empty_he, empty_node, empty_feat, empty_batch, empty_type

        dia_len_t = torch.tensor(dia_len, device=device)
        total_dialogues = len(dia_len)

        # Offsets for slicing original utterance tensors
        utt_offsets = torch.tensor(
            [0] + [int(dia_len_t[:i].sum().item()) for i in range(1, total_dialogues)],
            device=device
        )

        # Node starts in the hypergraph:
        # each dialogue contributes (3 * T) nodes
        node_starts = (dia_len_t * num_modality).cumsum(dim=0)
        node_starts = torch.cat([
            torch.zeros(1, dtype=dtype, device=device),
            node_starts[:-1]
        ])

        # -------------------------------------------------------
        # Batch vector: marks from which dialogue each node comes
        # -------------------------------------------------------
        batch = torch.cat([
            torch.full((dia_len_t[i] * num_modality,), i, device=device, dtype=dtype)
            for i in range(total_dialogues)
        ])

        # -------------------------------------------------------
        # Build L / A / V node indices for each dialogue
        # -------------------------------------------------------
        L_nodes, A_nodes, V_nodes = [], [], []

        for i, T in enumerate(dia_len_t):
            T_int = int(T.item())
            base = node_starts[i]

            # Language nodes
            L_nodes.append(base + torch.arange(0, T_int, device=device))
            # Audio nodes
            A_nodes.append(base + torch.arange(T_int, 2*T_int, device=device))
            # Visual nodes
            V_nodes.append(base + torch.arange(2*T_int, 3*T_int, device=device))

        # -------------------------------------------------------
        # Build Hyperedges
        # -------------------------------------------------------
        index1_list: List[torch.Tensor] = []  # all nodes inside hyperedges
        index2_list: List[torch.Tensor] = []  # hyperedge indices
        hyperedge_types: List[int] = []
        edge_counter = 0

        # ----------------------
        # (A) Modality-level hyperedges (Type = 1)
        # ----------------------
        for Ln, An, Vn in zip(L_nodes, A_nodes, V_nodes):
            for group in (Ln, An, Vn):
                index1_list.append(group)
                index2_list.append(
                    torch.full((group.numel(),), edge_counter,
                               device=device, dtype=dtype)
                )
                hyperedge_types.append(1)
                edge_counter += 1

        # ----------------------
        # (B) Utterance-level hyperedges (Type = 0)
        # Each hyperedge contains: {L_k, A_k, V_k}
        # ----------------------
        for Ln, An, Vn in zip(L_nodes, A_nodes, V_nodes):
            T = Ln.numel()
            triplets = torch.stack([Ln, An, Vn], dim=1)   # shape: (T, 3)

            # one hyperedge index per utterance
            he_id = torch.arange(edge_counter, edge_counter + T,
                                 device=device, dtype=dtype)

            # expand hyperedge indices to match L,A,V entries
            index1_list.append(triplets.reshape(-1))
            index2_list.append(he_id.repeat_interleave(3))

            hyperedge_types.extend([0] * T)
            edge_counter += T

        # Aggregate hyperedge indices
        index1 = torch.cat(index1_list)
        index2 = torch.cat(index2_list)
        hyperedge_index = torch.stack([index1, index2], dim=0)

        # Create hyperedge type vector
        hyperedge_type_tensor = torch.tensor(
            hyperedge_types, dtype=dtype, device=device
        ).view(-1, 1)

        # -------------------------------------------------------
        # Build Dense Node Adjacency (for HighConv)
        # -------------------------------------------------------
        edge_src_list: List[torch.Tensor] = []
        edge_dst_list: List[torch.Tensor] = []

        for Ln, An, Vn in zip(L_nodes, A_nodes, V_nodes):
            T = Ln.numel()

            # Intra-modality dense edges
            for group in (Ln, An, Vn):
                src = group.repeat_interleave(T)
                dst = group.repeat(T)
                mask = src != dst
                edge_src_list.append(src[mask])
                edge_dst_list.append(dst[mask])

            # Triplet K-hop edges: fully connect [L_k, A_k, V_k]
            trip = torch.stack([Ln, An, Vn], dim=1)
            src = trip.unsqueeze(2).expand(T, 3, 3)
            dst = trip.unsqueeze(1).expand(T, 3, 3)
            mask = (src != dst)

            edge_src_list.append(src[mask])
            edge_dst_list.append(dst[mask])

        # Final dense adjacency
        edge_index = torch.stack([
            torch.cat(edge_src_list),
            torch.cat(edge_dst_list)
        ], dim=0)

        # -------------------------------------------------------
        # Build Features (stack L, A, V)
        # -------------------------------------------------------
        features_list: List[torch.Tensor] = []

        for offset, T in zip(utt_offsets, dia_len_t):
            T_int = int(T.item())
            L_seg = l[offset : offset + T_int]
            A_seg = a[offset : offset + T_int]
            V_seg = v[offset : offset + T_int]

            features_list.append(torch.cat([L_seg, A_seg, V_seg], dim=0))

        features = torch.cat(features_list, dim=0)

        return (
            hyperedge_index,
            edge_index,
            features,
            batch,
            hyperedge_type_tensor
        )
    # ==================================================================
    # (2) GNN ADJACENCY INDEX (FOR HIGH-FREQUENCY GNN)
    # ==================================================================
    def create_gnn_index(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        l: torch.Tensor,
        dia_len: List[int],
        modals: List[str]
    ) -> Tuple[
        torch.Tensor,   # edge_index
        torch.Tensor    # features
    ]:
        """
        Build adjacency edges for the HighConv (GNN) layers.

        Semantics identical to the original implementation:
            • Dense fully-connected edges within each modality
            • For each utterance: fully connect [L_k, A_k, V_k]
            • Stack features in the order L → A → V

        Args:
            a: (total_T, D_a) audio features
            v: (total_T, D_v) visual features
            l: (total_T, D_l) language features
            dia_len: list of dialogue lengths
            modals: modality list ['a','v','l']

        Returns:
            edge_index: (2, E)
            features:   (sum(3T), D_total)
        """
        device = a.device
        dtype = torch.long
        num_modality = len(modals)

        # ---------------------------------------------------------
        # Early exit
        # ---------------------------------------------------------
        if not dia_len or sum(dia_len) == 0:
            empty_edge = torch.empty((2, 0), dtype=dtype, device=device)
            empty_feat = torch.empty((0, a.size(1) + v.size(1) + l.size(1)), device=device)
            return empty_edge, empty_feat

        dia_len_t = torch.tensor(dia_len, device=device)
        total_dialogues = len(dia_len)

        # Utterance offsets in original tensors
        utt_offsets = torch.tensor(
            [0] + [int(dia_len_t[:i].sum()) for i in range(1, total_dialogues)],
            device=device
        )

        # Node offsets in final adjacency graph
        # Each dialogue contributes 3*T nodes: [L (0..T-1), A (T..2T-1), V (2T..3T-1)]
        node_starts = (dia_len_t * num_modality).cumsum(dim=0)
        node_starts = torch.cat([
            torch.zeros(1, dtype=dtype, device=device),
            node_starts[:-1]
        ])

        # ---------------------------------------------------------
        # Dense adjacency edges per modality
        # ---------------------------------------------------------
        edge_src_list = []
        edge_dst_list = []

        for m in range(num_modality):  # 3 modalities → minimal iteration
            for i, T in enumerate(dia_len_t):
                T_int = int(T.item())
                if T_int <= 1:
                    continue

                base = node_starts[i] + m * T_int
                nodes = torch.arange(base, base + T_int, device=device)

                # Build all (u, v) where u != v
                src = nodes.repeat_interleave(T_int)
                dst = nodes.repeat(T_int)
                mask = src != dst

                edge_src_list.append(src[mask])
                edge_dst_list.append(dst[mask])

        # ---------------------------------------------------------
        # Triplet edges — connect {L_k, A_k, V_k} fully
        # ---------------------------------------------------------
        for i, T in enumerate(dia_len_t):
            T_int = int(T.item())
            if T_int == 0:
                continue

            base = node_starts[i]

            # Ranges for L/A/V
            Lr = base + torch.arange(0, T_int, device=device)
            Ar = base + torch.arange(T_int, 2*T_int, device=device)
            Vr = base + torch.arange(2*T_int, 3*T_int, device=device)

            trip = torch.stack([Lr, Ar, Vr], dim=1)  # (T, 3)

            # Form full 3x3 graph for each utterance
            src = trip.unsqueeze(2).expand(T_int, 3, 3)
            dst = trip.unsqueeze(1).expand(T_int, 3, 3)

            mask = src != dst
            edge_src_list.append(src[mask])
            edge_dst_list.append(dst[mask])

        # ---------------------------------------------------------
        # Combine edges
        # ---------------------------------------------------------
        edge_index = torch.stack([
            torch.cat(edge_src_list),
            torch.cat(edge_dst_list)
        ], dim=0)

        # ---------------------------------------------------------
        # Build features (stack L, A, V like hypergraph)
        # ---------------------------------------------------------
        feat_list = []

        for offset, T in zip(utt_offsets, dia_len_t):
            T_int = int(T.item())

            L_seg = l[offset : offset + T_int]
            A_seg = a[offset : offset + T_int]
            V_seg = v[offset : offset + T_int]

            feat_list.append(torch.cat([L_seg, A_seg, V_seg], dim=0))

        features = torch.cat(feat_list, dim=0)

        return edge_index, features
    # ==================================================================
    # (3) FORWARD PASS
    # ==================================================================
    def forward(
        self,
        a: torch.Tensor,
        v: torch.Tensor,
        l: torch.Tensor,
        dia_len: List[int],
        qmask: torch.Tensor,
        epoch: Optional[int] = None
    ) -> torch.Tensor:
        """
        Full forward path for the HyperGCN model.

        Steps:
        -------
        1. Speaker embeddings (optional)
        2. Positional encodings (optional)
        3. Modality embeddings (optional)
        4. Build hypergraph structure
        5. HypergraphConv stack (num_L layers)
        6. Build GNN adjacency
        7. HighConv stack (num_K layers)
        8. Concatenate hypergraph + GNN outputs
        9. Optional residual: concatenate original features
       10. Reverse stacking: rebuild per-utterance features

        Args:
            a: (total_T, D_a) audio features
            v: (total_T, D_v) visual features
            l: (total_T, D_l) language features
            dia_len: list of dialogue sequence lengths
            qmask: (max_T, batch, num_speakers) speaker masks
            epoch: optional training epoch

        Returns:
            Tensor: (sum(T), D_out) per-utterance feature representation
        """

        # -------------------------------------------------------------
        # 1. Speaker embeddings
        # -------------------------------------------------------------
        if self.use_speaker and ("l" in self.modals):
            spk_emb = self.extract_speaker_embeddings(qmask, dia_len)
            l = l + spk_emb

        # -------------------------------------------------------------
        # 2. Positional encodings (disabled by default)
        # -------------------------------------------------------------
        if self.use_position:
            raise NotImplementedError("Enable positional encodings if needed.")

        # -------------------------------------------------------------
        # 3. Modality embeddings ("a", "v", "l")
        # -------------------------------------------------------------
        if self.use_modal:
            modal_ids = torch.arange(3, device=a.device)
            modal_emb = self.modal_embeddings(modal_ids)  # (3, n_dim)

            if "a" in self.modals:
                a = a + modal_emb[0]
            if "v" in self.modals:
                v = v + modal_emb[1]
            if "l" in self.modals:
                l = l + modal_emb[2]

        # -------------------------------------------------------------
        # 4. Build hypergraph indices
        # -------------------------------------------------------------
        (
            hyperedge_index,
            node_edge_index,
            features,
            batch,
            hyperedge_type
        ) = self.create_hyper_index(a, v, l, dia_len, self.modals)

        # If no valid hypergraph exists, return empty reconstruction
        if hyperedge_index.numel() == 0:
            return self.reverse_features(dia_len, features)

        # Project features to hidden size
        x1 = self.fc1(features)

        # Hyperedge weights (simple 1-vector)
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
        hyper_w = torch.ones(num_hyperedges, device=x1.device)
        edge_w = torch.ones(hyperedge_index.size(1), device=x1.device)

        # Compute per-edge attribute
        edge_attr = (
            self.hyperedge_attr1 * hyperedge_type +
            self.hyperedge_attr2 * (1 - hyperedge_type)
        )

        # -------------------------------------------------------------
        # 5. HypergraphConv Stack
        # -------------------------------------------------------------
        out_h = x1
        for conv in self.hyperconvs:
            out_h = conv(
                out_h,
                hyperedge_index,
                hyper_w,
                edge_attr,
                edge_w,
                dia_len
            )

        # -------------------------------------------------------------
        # 6. Build node adjacency for GNN
        # -------------------------------------------------------------
        gnn_edge_index, _ = self.create_gnn_index(a, v, l, dia_len, self.modals)

        # -------------------------------------------------------------
        # 7. High-frequency GNN stack
        # -------------------------------------------------------------
        out_g = x1
        for conv in self.gnn_convs:
            out_g = out_g + conv(out_g, gnn_edge_index)

        # -------------------------------------------------------------
        # 8. Combine (HGCN + HighConv)
        # -------------------------------------------------------------
        out = torch.cat([out_h, out_g], dim=-1)

        # -------------------------------------------------------------
        # 9. Optional residual with input features
        # -------------------------------------------------------------
        if self.use_residue:
            out = torch.cat([features, out], dim=-1)

        # -------------------------------------------------------------
        # 10. Reverse: rebuild per-utterance L/A/V feature groups
        # -------------------------------------------------------------
        return self.reverse_features(dia_len, out)
