"""
Bipartite-tailored GNN encoders for MILP constraint-variable graphs.

Unlike generic GNNs, every encoder here respects the MILP bipartite structure:

  * messages are passed by *node type* in two interleaved half-convolutions
    (variable -> constraint, then constraint -> variable), each with its own
    parameters (Gasse et al., NeurIPS 2019);
  * the real coefficient a_ij is consumed as an edge feature (see
    src.utils.utils.build_edge_features), not discarded;
  * attention/normalization, when used, is restricted to bipartite neighbours
    (per-constraint over its variables = "intra-constraint competition").

All policies share the Predict-and-Search forward signature

    forward(constraint_features, edge_indices, edge_features,
            variable_features, batch_indices=None, is_training=False)
        -> per-variable logits  (shape [n_var])

so they are drop-in replacements for GNNPolicy in the PS-family trainer /
evaluator, selected by `gnn_type`.

References
----------
- Gasse, Chételat, Ferroni, Charlin, Lodi. "Exact Combinatorial Optimization
  with Graph Convolutional Neural Networks." NeurIPS 2019.
- Chen, Liu, Wang, Lu, Yin. "On Representing Mixed-Integer Linear Programs by
  Graph Neural Networks." ICLR 2023.  (random-feature augmentation)
- Ding, Zhang, Shen, Li, Wang, Xu, Song. "Accelerating Primal Solution Findings
  ... Based on Solution Prediction." AAAI 2020.  (tripartite objective node)
- Ye, Xu, Wang et al. "GNN&GBDT-Guided Fast Optimizing Framework ..." ICML 2023.
"""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import softmax

from .gcn import BipartiteGraphConvolution


# --------------------------------------------------------------------------- #
#  Shared feature embedding for the two node types + edges
# --------------------------------------------------------------------------- #
class _NodeEmbedding(nn.Module):
    def __init__(self, cons_nfeats, edge_nfeats, var_nfeats, emb_size):
        super().__init__()
        self.cons_embedding = nn.Sequential(
            nn.LayerNorm(cons_nfeats),
            nn.Linear(cons_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(nn.LayerNorm(edge_nfeats))
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(var_nfeats),
            nn.Linear(var_nfeats, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )

    def forward(self, c, e, v):
        return self.cons_embedding(c), self.edge_embedding(e), self.var_embedding(v)


# =========================================================================== #
#  1. Gasse half-convolution (configurable depth, jumping-knowledge, edge=a_ij)
# =========================================================================== #
class GasseGNNPolicy(nn.Module):
    """
    Deep Gasse-style half-convolution network. Identical message scheme to the
    default GNNPolicy but with configurable depth and jumping-knowledge (concat
    of every layer's variable embedding) before the read-out MLP.
    """

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.embed = _NodeEmbedding(constraint_nfeats, edge_nfeats, variable_nfeats, emb_size)

        self.v_to_c = nn.ModuleList(
            [BipartiteGraphConvolution(emb_size, edge_nfeats=edge_nfeats) for _ in range(depth)]
        )
        self.c_to_v = nn.ModuleList(
            [BipartiteGraphConvolution(emb_size, edge_nfeats=edge_nfeats) for _ in range(depth)]
        )

        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c, e, v = self.embed(constraint_features, edge_features, variable_features)

        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)
            v = self.c_to_v[i](c, edge_indices, e, v)
            v_hist.append(v)

        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)


# =========================================================================== #
#  2. Edge-featured bipartite attention (intra-constraint competition)
# =========================================================================== #
class BipartiteAttentionConv(torch_geometric.nn.MessagePassing):
    """
    GATv2-style attention where the softmax is taken over each *target* node's
    bipartite neighbourhood, and the edge coefficient a_ij modulates the
    attention logit. For the variable->constraint pass this makes the variables
    of one constraint compete for that constraint's attention (the origin of
    "intra-constraint competition"); the constraint->variable pass is symmetric.
    """

    def __init__(self, emb_size=64, edge_emb=64, heads=4, dropout=0.0):
        super().__init__(aggr="add", node_dim=0)
        assert emb_size % heads == 0, "emb_size must be divisible by heads"
        self.emb_size = emb_size
        self.heads = heads
        self.hd = emb_size // heads
        self.dropout = dropout

        self.lin_src = nn.Linear(emb_size, emb_size)
        self.lin_dst = nn.Linear(emb_size, emb_size)
        self.lin_edge = nn.Linear(edge_emb, emb_size, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, self.hd))
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.out = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        nn.init.xavier_uniform_(self.att)

    def forward(self, left_features, edge_indices, edge_features, right_features):
        agg = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            x=(left_features, right_features),
            edge_features=edge_features,
        )
        agg = self.update_mlp(agg)
        return self.out(torch.cat([agg, right_features], dim=-1))

    def message(self, x_j, x_i, edge_features, index, size_i):
        H, D = self.heads, self.hd
        src = self.lin_src(x_j).view(-1, H, D)
        dst = self.lin_dst(x_i).view(-1, H, D)
        edge = self.lin_edge(edge_features).view(-1, H, D)
        e = F.leaky_relu(src + dst + edge, 0.2)
        alpha = (e * self.att).sum(dim=-1)               # [E, H]
        alpha = softmax(alpha, index, num_nodes=size_i)  # per target node
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return ((src + edge) * alpha.unsqueeze(-1)).view(-1, self.emb_size)


class BipartiteAttentionPolicy(nn.Module):
    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=3, heads=4, dropout=0.0,
                 jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.embed = _NodeEmbedding(constraint_nfeats, edge_nfeats, variable_nfeats, emb_size)
        self.v_to_c = nn.ModuleList(
            [BipartiteAttentionConv(emb_size, edge_nfeats, heads, dropout) for _ in range(depth)]
        )
        self.c_to_v = nn.ModuleList(
            [BipartiteAttentionConv(emb_size, edge_nfeats, heads, dropout) for _ in range(depth)]
        )
        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c, e, v = self.embed(constraint_features, edge_features, variable_features)
        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)
            v = self.c_to_v[i](c, edge_indices, e, v)
            v_hist.append(v)
        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)


# =========================================================================== #
#  3. Random-feature augmented Gasse (Chen et al., ICLR 2023 expressiveness fix)
# =========================================================================== #
class RandomFeatureGNNPolicy(nn.Module):
    """
    Gasse backbone with `n_rand` i.i.d. random features appended to every
    variable node, resampled on each forward pass. Chen et al. (ICLR 2023) prove
    plain message-passing GNNs cannot separate some non-isomorphic MILPs
    ("foldable" instances); appending random node features provably restores
    separation power (w.h.p.).
    """

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, n_rand=8, jumping_knowledge=True,
                 n_eval_samples=8):
        super().__init__()
        self.n_rand = n_rand
        # at eval, average this many random-feature draws for a stable prediction
        self.n_eval_samples = n_eval_samples
        self.backbone = GasseGNNPolicy(
            emb_size=emb_size,
            constraint_nfeats=constraint_nfeats,
            edge_nfeats=edge_nfeats,
            variable_nfeats=variable_nfeats + n_rand,
            depth=depth,
            jumping_knowledge=jumping_knowledge,
        )

    def _one_pass(self, constraint_features, edge_indices, edge_features,
                  variable_features, batch_indices, is_training):
        n = variable_features.shape[0]
        rand = torch.randn(n, self.n_rand, device=variable_features.device,
                           dtype=variable_features.dtype)
        v_aug = torch.cat([variable_features, rand], dim=-1)
        return self.backbone(constraint_features, edge_indices, edge_features,
                             v_aug, batch_indices, is_training)

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        if self.training:
            # single stochastic draw per step during training
            return self._one_pass(constraint_features, edge_indices, edge_features,
                                  variable_features, batch_indices, is_training)
        # deterministic-in-expectation: average several random-feature draws
        logits = 0.0
        for _ in range(max(1, self.n_eval_samples)):
            logits = logits + self._one_pass(constraint_features, edge_indices,
                                             edge_features, variable_features,
                                             batch_indices, is_training)
        return logits / max(1, self.n_eval_samples)


# =========================================================================== #
#  4. Tripartite: explicit objective node (Ding AAAI 2020 / Ye ICML 2023)
# =========================================================================== #
class TripartiteGNNPolicy(nn.Module):
    """
    Adds a single OBJECTIVE node connected to every variable, with edge weight =
    the variable's objective coefficient. The objective information therefore
    flows through dedicated message passing instead of being buried in a scalar
    node feature. One round = v->c, c->v, then v->o, o->v (four type-specific
    half-convolutions with independent parameters).

    The objective coefficient is read from variable_features[:, obj_feat_idx]
    (column 0 is the normalized objective coefficient in the PS feature layout).
    """

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=3, obj_feat_idx=0, jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.obj_feat_idx = obj_feat_idx
        self.embed = _NodeEmbedding(constraint_nfeats, edge_nfeats, variable_nfeats, emb_size)
        # objective node starts from a learned embedding
        self.obj_init = nn.Parameter(torch.zeros(1, emb_size))
        # objective edges carry a 1-dim feature = normalized objective coefficient
        self.v_to_c = nn.ModuleList([BipartiteGraphConvolution(emb_size, edge_nfeats) for _ in range(depth)])
        self.c_to_v = nn.ModuleList([BipartiteGraphConvolution(emb_size, edge_nfeats) for _ in range(depth)])
        self.v_to_o = nn.ModuleList([BipartiteGraphConvolution(emb_size, 1) for _ in range(depth)])
        self.o_to_v = nn.ModuleList([BipartiteGraphConvolution(emb_size, 1) for _ in range(depth)])

        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        device = variable_features.device
        n_var = variable_features.shape[0]
        if batch_indices is None:
            batch_indices = torch.zeros(n_var, dtype=torch.long, device=device)
        num_graphs = int(batch_indices.max()) + 1

        # One objective node PER GRAPH; variable j connects to its graph's obj
        # node (batch_indices[j]). Prevents cross-graph leakage in batched runs.
        obj_coeff = variable_features[:, self.obj_feat_idx]
        var_ids = torch.arange(n_var, device=device)
        obj_edge_index = torch.stack([batch_indices, var_ids], dim=0)   # (obj, var)
        rev_obj = torch.stack([var_ids, batch_indices], dim=0)          # (var, obj)
        obj_edge_feat = obj_coeff.reshape(-1, 1)                        # 1-dim coeff

        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c, e, v = self.embed(constraint_features, edge_features, variable_features)
        o = self.obj_init.expand(num_graphs, -1).contiguous()

        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)
            v = self.c_to_v[i](c, edge_indices, e, v)
            # variable -> objective (aggregate each graph's variables into its obj node)
            o = self.v_to_o[i](v, rev_obj, obj_edge_feat, o)
            # objective -> variable (broadcast the per-graph objective context back)
            v = self.o_to_v[i](o, obj_edge_index, obj_edge_feat, v)
            v_hist.append(v)

        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)


# =========================================================================== #
#  5. Global-attention graph transformer (WL-characterized, arXiv 2607.17570)
# =========================================================================== #
class GraphTransformerPolicy(nn.Module):
    """
    Combines local bipartite attention (structure-respecting half-convolutions)
    with a *global attention* block per layer, motivated by the WL
    characterization of global-attention graph transformers for MILP
    (arXiv 2607.17570). Global attention is realized with an attention-pooled
    global token per graph (O(n) instead of O(n^2)) that is broadcast back to
    every variable, letting distant variables exchange information in one hop.
    """

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=3, heads=4, dropout=0.0,
                 jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.embed = _NodeEmbedding(constraint_nfeats, edge_nfeats, variable_nfeats, emb_size)
        self.v_to_c = nn.ModuleList(
            [BipartiteAttentionConv(emb_size, edge_nfeats, heads, dropout) for _ in range(depth)]
        )
        self.c_to_v = nn.ModuleList(
            [BipartiteAttentionConv(emb_size, edge_nfeats, heads, dropout) for _ in range(depth)]
        )
        self.global_score = nn.ModuleList([nn.Linear(emb_size, 1) for _ in range(depth)])
        self.global_fuse = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * emb_size, emb_size), nn.ReLU(),
                           nn.Linear(emb_size, emb_size)) for _ in range(depth)]
        )
        self.global_norm = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(depth)])
        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size), nn.ReLU(), nn.Linear(emb_size, 1, bias=False),
        )

    def _global(self, v, batch_indices, i):
        # attention pooling over variables per graph -> global token, broadcast back
        alpha = softmax(self.global_score[i](v).squeeze(-1), batch_indices)   # [n_var]
        num_graphs = int(batch_indices.max()) + 1
        g = torch.zeros(num_graphs, v.shape[1], device=v.device).index_add(
            0, batch_indices, v * alpha.unsqueeze(-1))
        fused = self.global_fuse[i](torch.cat([v, g[batch_indices]], dim=-1))
        return self.global_norm[i](v + fused)

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        n_var = variable_features.shape[0]
        if batch_indices is None:
            batch_indices = torch.zeros(n_var, dtype=torch.long, device=variable_features.device)
        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c, e, v = self.embed(constraint_features, edge_features, variable_features)
        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)
            v = self.c_to_v[i](c, edge_indices, e, v)
            v = self._global(v, batch_indices, i)          # global attention block
            v_hist.append(v)
        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)


# =========================================================================== #
# =========================================================================== #
#  Higher-expressive-power encoders for the MILP bipartite graph.
#
#     Predict-and-Search predicts a per-binary-variable marginal and searches a
#     trust region around it. If the encoder cannot tell two variables apart it
#     must give them the same marginal, so prediction quality is capped by the
#     encoder's ability to separate nodes -- its expressive power, formalized
#     through the Weisfeiler-Leman (WL) hierarchy.
#
#     References
#     ----------
#     * Chen, Liu, Wang, Lu, Yin. "On Representing Mixed-Integer Linear Programs
#       by Graph Neural Networks." ICLR 2023 (arXiv:2210.10759). Message-passing
#       GNNs on the constraint-variable graph are bounded by the 1-WL test, so
#       there exist non-isomorphic ("foldable") MILPs that every such GNN maps to
#       identical embeddings; random node features restore separation power.
#     * Jahin, Knoblock, Pujara. "A Weisfeiler-Leman Characterization of
#       Global-Attention Graph Transformers for MILPs." arXiv:2607.17570 (2026).
#       Global attention is also 1-WL-bounded; power beyond 1-WL comes from the
#       input encoding -- specifically random-walk positional encodings.
#     * "Branching Strategies Based on Subgraph GNNs: Theoretical Promise vs
#       Practical Reality." arXiv:2512.09355 (2025). Subgraph / higher-order GNNs
#       are more expressive but pay an O(n) cost that often outweighs the gains.
#     * Xu, Hu, Leskovec, Jegelka. "How Powerful are Graph Neural Networks?"
#       ICLR 2019 (arXiv:1810.00826). Sum-aggregation GIN is the maximally
#       expressive MPNN (== 1-WL). GINE (Hu et al. 2020) adds edge features.
#     * Bouritsas, Frasca, Zafeiriou, Bronstein. "Improving GNN Expressivity via
#       Subgraph Isomorphism Counting" (GSN), TPAMI 2022. Appending substructure
#       counts yields provably-beyond-1-WL power.
#
#     Bipartite tailoring: on a constraint-variable graph the shortest cycle is a
#     4-cycle (variable-constraint-variable-constraint), so every odd-length
#     closed walk vanishes and the even-length return probabilities / counts are
#     exactly what encodes the girth-4 structure 1-WL cannot see. The structural
#     encoders exploit this (odd powers are ~0 by construction).
# =========================================================================== #

# ---- shared structural-encoding utilities (cached per instance) ----------- #
_SE_CACHE = {}
_SE_CACHE_ORDER = []
_SE_CACHE_MAX = 4096
# fill-in guard: stop the sparse power iteration once a power matrix exceeds this
# many non-zeros (dense fold-in on high-degree constraints would otherwise OOM);
# remaining powers are left at 0. Structural encodings are truncated, not wrong.
_SE_NNZ_CAP = 8_000_000


def _se_cache_put(key, value):
    if key in _SE_CACHE:
        return
    _SE_CACHE[key] = value
    _SE_CACHE_ORDER.append(key)
    if len(_SE_CACHE_ORDER) > _SE_CACHE_MAX:
        old = _SE_CACHE_ORDER.pop(0)
        _SE_CACHE.pop(old, None)


def _se_cache_key(edge_index, num_cons, num_var, tag):
    ei = edge_index.detach().to("cpu").contiguous()
    h = hashlib.blake2b(ei.numpy().tobytes(), digest_size=16).hexdigest()
    return (tag, int(num_cons), int(num_var), int(ei.shape[1]), h)


def _bipartite_sym_sparse(edge_index, num_cons, num_var):
    """Undirected, UNWEIGHTED adjacency of the whole bipartite graph as one
    N x N sparse matrix (N = num_cons + num_var). Constraint node i -> index i;
    variable node j -> index num_cons + j. Batched graphs are block-diagonal in
    edge_index (PyG offsets per graph), so operating on the full matrix is
    exactly per-graph -- no cross-graph leakage."""
    N = num_cons + num_var
    ci = edge_index[0].long().cpu()
    vi = edge_index[1].long().cpu() + num_cons
    src = torch.cat([ci, vi])
    dst = torch.cat([vi, ci])
    idx = torch.stack([src, dst])
    A = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1]), (N, N)).coalesce()
    # unweighted structure: clamp any merged multi-edges back to 1
    A = torch.sparse_coo_tensor(A.indices(), torch.ones_like(A.values()), (N, N)).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    return A, deg, N


def _sparse_diag(M, N):
    M = M.coalesce()
    ii, vv = M.indices(), M.values()
    m = ii[0] == ii[1]
    d = torch.zeros(N)
    d.index_add_(0, ii[0][m], vv[m])
    return d


def _rw_se_single(edge_index, num_cons, num_var, walk_length):
    """Single-instance RWSE core (see random_walk_se). Cached per instance."""
    key = _se_cache_key(edge_index, num_cons, num_var, ("rwse", int(walk_length)))
    hit = _SE_CACHE.get(key)
    if hit is not None:
        return hit
    with torch.no_grad():
        A, deg, N = _bipartite_sym_sparse(edge_index, num_cons, num_var)
        dinv = torch.where(deg > 0, 1.0 / deg, torch.zeros_like(deg))
        ai, av = A.indices(), A.values()
        P = torch.sparse_coo_tensor(ai, av * dinv[ai[0]], (N, N)).coalesce()
        se = torch.zeros(N, walk_length)
        Pk = P
        for k in range(walk_length):
            se[:, k] = _sparse_diag(Pk, N)
            if k < walk_length - 1:
                if Pk._nnz() > _SE_NNZ_CAP:
                    break  # per-instance fill-in guard: leave deeper walks at 0
                Pk = torch.sparse.mm(Pk, P)
    _se_cache_put(key, se)
    return se


def _struct_counts_single(edge_index, num_cons, num_var, powers):
    """Single-instance structural-count core (see bipartite_structural_counts).
    Cached per instance."""
    key = _se_cache_key(edge_index, num_cons, num_var, ("sub", tuple(powers)))
    hit = _SE_CACHE.get(key)
    if hit is not None:
        return hit
    with torch.no_grad():
        A, deg, N = _bipartite_sym_sparse(edge_index, num_cons, num_var)
        ai = A.indices()
        nbr_deg_sum = torch.zeros(N).index_add_(0, ai[0], deg[ai[1]])
        nbr_deg_mean = torch.where(deg > 0, nbr_deg_sum / deg.clamp(min=1), torch.zeros_like(deg))
        nbr_deg_max = torch.zeros(N).scatter_reduce(
            0, ai[0], deg[ai[1]], reduce="amax", include_self=False
        )
        maxp = max(powers)
        diags = {}
        cur = A
        for k in range(2, maxp + 1):
            if cur._nnz() > _SE_NNZ_CAP:
                break  # per-instance fill-in guard: remaining powers stay 0
            cur = torch.sparse.mm(cur, A).coalesce()
            if k in powers:
                diags[k] = _sparse_diag(cur, N)
        cols = [torch.log1p(deg), torch.log1p(nbr_deg_mean), torch.log1p(nbr_deg_max)]
        for p in powers:
            cols.append(torch.log1p(diags.get(p, torch.zeros(N))))
        se = torch.stack(cols, dim=1)
    _se_cache_put(key, se)
    return se


def _per_graph_se(core_fn, width, edge_index, num_cons, num_var, batch_indices, *core_args):
    """Apply a single-instance structural-encoding core PER GRAPH and reassemble
    into the batched [num_cons + num_var, width] layout (constraint rows first).

    This matters for two reasons flagged in review:
      * the sparse-power fill-in guard (_SE_NNZ_CAP) must see one instance's
        nnz, not the SUM over a block-diagonal batch (which would truncate
        encodings differently depending on batch size);
      * the per-instance cache key is stable across epochs/shuffles, so shuffled
        mini-batches still hit the cache.
    With a single graph (batch_indices None or all-zero) this is a no-op wrapper
    around core_fn on the full graph."""
    if batch_indices is None or int(batch_indices.max()) == 0:
        return core_fn(edge_index, num_cons, num_var, *core_args)

    device = edge_index.device
    num_graphs = int(batch_indices.max()) + 1
    # every edge of a constraint belongs to one graph; read it off its variable
    cons_batch = torch.full((num_cons,), -1, dtype=torch.long, device=device)
    cons_batch[edge_index[0]] = batch_indices[edge_index[1]]

    se_cons = torch.zeros(num_cons, width)
    se_var = torch.zeros(num_var, width)
    for g in range(num_graphs):
        c_ids = (cons_batch == g).nonzero(as_tuple=False).flatten()
        v_ids = (batch_indices == g).nonzero(as_tuple=False).flatten()
        emask = batch_indices[edge_index[1]] == g
        sub_ei = edge_index[:, emask]
        cons_remap = torch.full((num_cons,), -1, dtype=torch.long, device=device)
        cons_remap[c_ids] = torch.arange(c_ids.numel(), device=device)
        var_remap = torch.full((num_var,), -1, dtype=torch.long, device=device)
        var_remap[v_ids] = torch.arange(v_ids.numel(), device=device)
        local_ei = torch.stack([cons_remap[sub_ei[0]], var_remap[sub_ei[1]]], dim=0)
        se_local = core_fn(local_ei, c_ids.numel(), v_ids.numel(), *core_args)
        nc = c_ids.numel()
        se_cons[c_ids.cpu()] = se_local[:nc]
        se_var[v_ids.cpu()] = se_local[nc:]
    return torch.cat([se_cons, se_var], dim=0)


def random_walk_se(edge_index, num_cons, num_var, walk_length, batch_indices=None):
    """Random-Walk Structural Encoding: for every node, the return probabilities
    diag(P^k), k = 1..walk_length, of the random walk P = D^{-1} A on the
    bipartite graph. Deterministic and permutation-equivariant. Returns a
    [num_cons + num_var, walk_length] tensor (constraint rows -- incl. the
    objective node -- first, then variables). Computed PER GRAPH (see
    _per_graph_se) and cached per instance."""
    return _per_graph_se(_rw_se_single, int(walk_length), edge_index,
                         num_cons, num_var, batch_indices, int(walk_length))


def bipartite_structural_counts(edge_index, num_cons, num_var, powers=(4, 6),
                                batch_indices=None):
    """GSN-style structural features per node:
        [ log1p(degree),
          log1p(mean neighbour degree),        (1-WL context)
          log1p(max  neighbour degree),        (1-WL context)
          log1p(diag(A^p)) for p in powers ]   (closed-walk / butterfly counts)
    On a bipartite graph diag(A^4) counts closed 4-walks == 4-cycles ("butterfly"
    energy) plus degeneracies -- an invariant 1-WL cannot compute. Computed PER
    GRAPH and cached per instance. Returns [num_cons + num_var, 3 + len(powers)]."""
    powers = tuple(powers)
    return _per_graph_se(_struct_counts_single, 3 + len(powers), edge_index,
                         num_cons, num_var, batch_indices, powers)


# =========================================================================== #
#  6a. Bipartite GINE (injective SUM aggregation == 1-WL ceiling)
# =========================================================================== #
class BipartiteGINEConv(torch_geometric.nn.MessagePassing):
    """GINE half-convolution (Hu et al. 2020). SUM aggregation with the (1 + eps)
    self-term is the GIN recipe that is as expressive as 1-WL (Xu et al. 2019) --
    strictly stronger than the mean/attention-style aggregators of the default
    encoders. The edge coefficient a_ij is folded in GINE-style: ReLU(x_j + W_e
    e_ij) before summation. A post-update norm is included for training stability,
    following the original GIN (which normalizes after the MLP)."""

    def __init__(self, emb_size=64, edge_nfeats=2, train_eps=True):
        super().__init__(aggr="add", node_dim=0)
        self.edge_lin = nn.Linear(edge_nfeats, emb_size)
        if train_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.LayerNorm(emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        e = self.edge_lin(edge_features)
        agg = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            x=(left_features, right_features),
            edge_emb=e,
        )
        return self.mlp((1.0 + self.eps) * right_features + agg)

    def message(self, x_j, edge_emb):
        return F.relu(x_j + edge_emb)


class BipartiteGINPolicy(nn.Module):
    """Deep bipartite GINE with jumping-knowledge. Same two-interleaved-half-
    convolution scheme as GasseGNNPolicy but with injective GINE updates, so it
    reaches the 1-WL ceiling the weaker gcn/attention encoders sit below."""

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, train_eps=True, jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.embed = _NodeEmbedding(constraint_nfeats, edge_nfeats, variable_nfeats, emb_size)
        self.v_to_c = nn.ModuleList(
            [BipartiteGINEConv(emb_size, edge_nfeats, train_eps) for _ in range(depth)]
        )
        self.c_to_v = nn.ModuleList(
            [BipartiteGINEConv(emb_size, edge_nfeats, train_eps) for _ in range(depth)]
        )
        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c, e, v = self.embed(constraint_features, edge_features, variable_features)
        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)
            v = self.c_to_v[i](c, edge_indices, e, v)
            v_hist.append(v)
        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)


# =========================================================================== #
#  6b. Random-Walk Structural Encoding + Gasse backbone (beyond 1-WL)
# =========================================================================== #
class RWSEGNNPolicy(nn.Module):
    """Gasse half-convolution backbone whose node inputs are augmented with a
    deterministic Random-Walk Structural Encoding (diag(P^k), k=1..walk_length)
    computed on the bipartite graph. Provably separates 1-WL-equivalent MILPs
    (arXiv:2607.17570) with far lower variance than random_feature, because the
    encoding is a fixed function of the graph rather than fresh noise."""

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, walk_length=8, jumping_knowledge=True):
        super().__init__()
        self.walk_length = walk_length
        self.backbone = GasseGNNPolicy(
            emb_size=emb_size,
            constraint_nfeats=constraint_nfeats + walk_length,
            edge_nfeats=edge_nfeats,
            variable_nfeats=variable_nfeats + walk_length,
            depth=depth,
            jumping_knowledge=jumping_knowledge,
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        num_cons = constraint_features.shape[0]
        num_var = variable_features.shape[0]
        se = random_walk_se(edge_indices, num_cons, num_var, self.walk_length,
                            batch_indices=batch_indices)
        se = se.to(constraint_features.device, constraint_features.dtype)
        cf = torch.cat([constraint_features, se[:num_cons]], dim=1)
        vf = torch.cat([variable_features, se[num_cons:]], dim=1)
        return self.backbone(cf, edge_indices, edge_features, vf, batch_indices, is_training)


# =========================================================================== #
#  6c. GSN-style bipartite substructure counts + Gasse backbone (beyond 1-WL)
# =========================================================================== #
class SubstructureGNNPolicy(nn.Module):
    """Gasse backbone whose node inputs are augmented with GSN-style bipartite
    motif counts (see bipartite_structural_counts): a local degree profile plus
    closed-walk / butterfly energy diag(A^p). The closed-walk counts are a
    beyond-1-WL invariant that directly measures 4-cycle participation, the
    fundamental motif of a constraint-variable graph."""

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, powers=(4, 6), jumping_knowledge=True):
        super().__init__()
        self.powers = tuple(powers)
        self.se_width = 3 + len(self.powers)
        self.backbone = GasseGNNPolicy(
            emb_size=emb_size,
            constraint_nfeats=constraint_nfeats + self.se_width,
            edge_nfeats=edge_nfeats,
            variable_nfeats=variable_nfeats + self.se_width,
            depth=depth,
            jumping_knowledge=jumping_knowledge,
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        num_cons = constraint_features.shape[0]
        num_var = variable_features.shape[0]
        se = bipartite_structural_counts(edge_indices, num_cons, num_var,
                                         self.powers, batch_indices=batch_indices)
        se = se.to(constraint_features.device, constraint_features.dtype)
        cf = torch.cat([constraint_features, se[:num_cons]], dim=1)
        vf = torch.cat([variable_features, se[num_cons:]], dim=1)
        return self.backbone(cf, edge_indices, edge_features, vf, batch_indices, is_training)


# =========================================================================== #
#  6d. Bipartite GraphGPS: local half-conv + global attention + RWSE (> 1-WL)
# =========================================================================== #
class GraphGPSPolicy(nn.Module):
    """Bipartite realization of GraphGPS (Rampasek et al., NeurIPS 2022): every
    layer combines a LOCAL message-passing half-convolution with a GLOBAL
    attention block, and the node inputs carry a Random-Walk Structural Encoding.
    Per arXiv:2607.17570 the attention itself is 1-WL-bounded; the RWSE input is
    what lifts the model BEYOND 1-WL. This differs from graph_transformer (local
    *attention* + global token, no PE, so 1-WL-bounded) precisely by that PE.
    """

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=3, heads=4, dropout=0.0,
                 walk_length=8, jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.walk_length = walk_length
        self.embed = _NodeEmbedding(constraint_nfeats + walk_length, edge_nfeats,
                                    variable_nfeats + walk_length, emb_size)
        # local message passing (MPNN half-convolutions)
        self.v_to_c = nn.ModuleList(
            [BipartiteGraphConvolution(emb_size, edge_nfeats=edge_nfeats) for _ in range(depth)]
        )
        self.c_to_v = nn.ModuleList(
            [BipartiteGraphConvolution(emb_size, edge_nfeats=edge_nfeats) for _ in range(depth)]
        )
        # global attention block (attention-pooled per-graph token, broadcast back)
        self.global_score = nn.ModuleList([nn.Linear(emb_size, 1) for _ in range(depth)])
        self.global_fuse = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * emb_size, emb_size), nn.ReLU(),
                           nn.Linear(emb_size, emb_size)) for _ in range(depth)]
        )
        self.global_norm = nn.ModuleList([nn.LayerNorm(emb_size) for _ in range(depth)])
        self.dropout = dropout
        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size), nn.ReLU(), nn.Linear(emb_size, 1, bias=False),
        )

    def _global(self, v, batch_indices, i):
        alpha = softmax(self.global_score[i](v).squeeze(-1), batch_indices)
        num_graphs = int(batch_indices.max()) + 1
        g = torch.zeros(num_graphs, v.shape[1], device=v.device, dtype=v.dtype).index_add(
            0, batch_indices, v * alpha.unsqueeze(-1))
        fused = self.global_fuse[i](torch.cat([v, g[batch_indices]], dim=-1))
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        return self.global_norm[i](v + fused)

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        num_cons = constraint_features.shape[0]
        num_var = variable_features.shape[0]
        if batch_indices is None:
            batch_indices = torch.zeros(num_var, dtype=torch.long,
                                        device=variable_features.device)
        se = random_walk_se(edge_indices, num_cons, num_var, self.walk_length,
                            batch_indices=batch_indices)
        se = se.to(constraint_features.device, constraint_features.dtype)
        cf = torch.cat([constraint_features, se[:num_cons]], dim=1)
        vf = torch.cat([variable_features, se[num_cons:]], dim=1)

        rev = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c, e, v = self.embed(cf, edge_features, vf)
        v_hist = [v]
        for i in range(self.depth):
            c = self.v_to_c[i](v, rev, e, c)          # local MPNN
            v = self.c_to_v[i](c, edge_indices, e, v)
            v = self._global(v, batch_indices, i)      # global attention
            v_hist.append(v)
        h = torch.cat(v_hist, dim=-1) if self.jk else v
        return self.output_module(h).squeeze(-1)


# =========================================================================== #
#  6e. Laplacian spectral positional encoding + Gasse backbone (> 1-WL)
# =========================================================================== #
def _lap_pe_single(edge_index, num_cons, num_var, pe_dim):
    """k = pe_dim smallest-eigenvalue eigenvectors of the symmetric-normalized
    Laplacian L = I - D^{-1/2} A D^{-1/2} of the bipartite graph, plus the
    eigenvalues themselves. Returns [N, 2*pe_dim] = [eigvecs | eigvals broadcast].
    Eigenvalues are sign/basis-invariant and their multiplicity at 0 counts
    connected components, so they separate e.g. a single cycle from a union of
    cycles. Eigenvectors use a max-abs-positive sign convention (sign ambiguity
    is further handled by train-time sign flipping in the policy; a fully
    invariant treatment would use SignNet, Lim et al. 2022). Cached per instance.
    """
    key = _se_cache_key(edge_index, num_cons, num_var, ("lap", int(pe_dim)))
    hit = _SE_CACHE.get(key)
    if hit is not None:
        return hit
    import numpy as np
    with torch.no_grad():
        A, deg, N = _bipartite_sym_sparse(edge_index, num_cons, num_var)
        dhalf = deg.clamp(min=1).pow(-0.5)
        dhalf[deg == 0] = 0.0
        ai, av = A.indices(), A.values()
        lval = -(dhalf[ai[0]] * av * dhalf[ai[1]])
        k = min(pe_dim, N)
        if N <= 2500:
            L = torch.sparse_coo_tensor(ai, lval, (N, N)).to_dense()
            L = L + torch.eye(N)
            evals, evecs = torch.linalg.eigh(L)
            vec = evecs[:, :k].clone()
            val = evals[:k].clone()
        else:
            try:
                from scipy.sparse import coo_matrix, identity
                from scipy.sparse.linalg import eigsh
                Lsp = coo_matrix((lval.numpy(), (ai[0].numpy(), ai[1].numpy())),
                                 shape=(N, N)) + identity(N)
                kk = min(pe_dim, N - 1)
                w, U = eigsh(Lsp.tocsr(), k=kk, which="SA")
                order = np.argsort(w)
                vec = torch.from_numpy(U[:, order]).float()
                val = torch.from_numpy(w[order]).float()
                k = kk
            except Exception:
                vec = torch.zeros(N, k)
                val = torch.zeros(k)
        # canonical sign: largest-magnitude entry positive
        for j in range(vec.shape[1]):
            idx = vec[:, j].abs().argmax()
            if vec[idx, j] < 0:
                vec[:, j] = -vec[:, j]
        if k < pe_dim:
            vec = torch.cat([vec, torch.zeros(N, pe_dim - k)], dim=1)
            val = torch.cat([val, torch.zeros(pe_dim - k)])
        se = torch.cat([vec, val.unsqueeze(0).expand(N, -1)], dim=1)
    _se_cache_put(key, se)
    return se


def laplacian_pe(edge_index, num_cons, num_var, pe_dim, batch_indices=None):
    """Per-graph Laplacian PE reassembled into [num_cons + num_var, 2*pe_dim]."""
    return _per_graph_se(_lap_pe_single, 2 * pe_dim, edge_index,
                         num_cons, num_var, batch_indices, pe_dim)


class LapPEGNNPolicy(nn.Module):
    """Gasse backbone whose node inputs carry a Laplacian spectral positional
    encoding (eigenvectors + eigenvalues; see _lap_pe_single). A spectral,
    deterministic route beyond 1-WL, complementary to the walk-based rwse. During
    training the eigenvector columns get random sign flips (sign augmentation)."""

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, pe_dim=8, jumping_knowledge=True):
        super().__init__()
        self.pe_dim = pe_dim
        self.backbone = GasseGNNPolicy(
            emb_size=emb_size,
            constraint_nfeats=constraint_nfeats + 2 * pe_dim,
            edge_nfeats=edge_nfeats,
            variable_nfeats=variable_nfeats + 2 * pe_dim,
            depth=depth,
            jumping_knowledge=jumping_knowledge,
        )

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        num_cons = constraint_features.shape[0]
        num_var = variable_features.shape[0]
        se = laplacian_pe(edge_indices, num_cons, num_var, self.pe_dim,
                          batch_indices=batch_indices)
        se = se.to(constraint_features.device, constraint_features.dtype)
        if self.training:  # sign-flip augmentation on the eigenvector columns only
            signs = (torch.randint(0, 2, (self.pe_dim,), device=se.device,
                                   dtype=se.dtype) * 2 - 1)
            se = se.clone()
            se[:, :self.pe_dim] = se[:, :self.pe_dim] * signs
        cf = torch.cat([constraint_features, se[:num_cons]], dim=1)
        vf = torch.cat([variable_features, se[num_cons:]], dim=1)
        return self.backbone(cf, edge_indices, edge_features, vf, batch_indices, is_training)


# =========================================================================== #
#  6f. Identity-aware GNN (You et al., ICLR 2021) -- subgraph-flavored, > 1-WL
# =========================================================================== #
def _graph_partition(edge_index, num_cons, num_var, batch_indices):
    """Yield (c_ids, v_ids, local_edge_index) for each graph in a (possibly
    batched, block-diagonal) bipartite graph. Isolated constraints are dropped
    from their graph's c_ids (they carry no edges)."""
    if batch_indices is None or int(batch_indices.max()) == 0:
        yield (torch.arange(num_cons, device=edge_index.device),
               torch.arange(num_var, device=edge_index.device), edge_index)
        return
    device = edge_index.device
    ng = int(batch_indices.max()) + 1
    cons_batch = torch.full((num_cons,), -1, dtype=torch.long, device=device)
    cons_batch[edge_index[0]] = batch_indices[edge_index[1]]
    for g in range(ng):
        c_ids = (cons_batch == g).nonzero(as_tuple=False).flatten()
        v_ids = (batch_indices == g).nonzero(as_tuple=False).flatten()
        emask = batch_indices[edge_index[1]] == g
        sub = edge_index[:, emask]
        cr = torch.full((num_cons,), -1, dtype=torch.long, device=device)
        cr[c_ids] = torch.arange(c_ids.numel(), device=device)
        vr = torch.full((num_var,), -1, dtype=torch.long, device=device)
        vr[v_ids] = torch.arange(v_ids.numel(), device=device)
        lei = torch.stack([cr[sub[0]], vr[sub[1]]], dim=0)
        yield (c_ids, v_ids, lei)


class IDGNNPolicy(nn.Module):
    """Identity-aware GNN (You, Gomes-Selman, Ying, Leskovec; ICLR 2021). For each
    ROOT variable it runs the shared half-convolution backbone with that root
    tagged by an extra identity feature, and reads the root's own logit from that
    tagged pass. Injecting and propagating the root's identity lets closed walks
    that return to the root be counted, which is provably BEYOND 1-WL (it
    separates 1-WL-equivalent regular graphs that plain / higher-order sum-MPNNs
    cannot).

    Cost is the honest catch (cf. arXiv:2512.09355): one backbone pass per root,
    i.e. O(n_var) passes. `max_roots` caps this -- when a graph has more binary
    variables than max_roots, only max_roots (evenly spaced) get the identity
    treatment and the rest fall back to the untagged base logit. Set max_roots
    >= n_var for the full, maximally expressive model.
    """

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=3, max_roots=32, jumping_knowledge=True):
        super().__init__()
        self.max_roots = max_roots
        self.backbone = GasseGNNPolicy(
            emb_size=emb_size,
            constraint_nfeats=constraint_nfeats,
            edge_nfeats=edge_nfeats,
            variable_nfeats=variable_nfeats + 1,  # +1 identity/marking channel
            depth=depth,
            jumping_knowledge=jumping_knowledge,
        )

    def _run_one_graph(self, cf, ei, ef, vf):
        """ID-GNN logits for a SINGLE graph (all-zero batch)."""
        n = vf.shape[0]
        device = vf.device
        zero = torch.zeros(n, 1, device=device, dtype=vf.dtype)
        base = self.backbone(cf, ei, ef, torch.cat([vf, zero], dim=1))  # untagged
        out = base.clone()
        if n <= self.max_roots:
            roots = torch.arange(n, device=device)
        else:
            roots = torch.linspace(0, n - 1, self.max_roots, device=device).round().long().unique()
        for r in roots.tolist():
            mark = torch.zeros(n, 1, device=device, dtype=vf.dtype)
            mark[r] = 1.0
            lg = self.backbone(cf, ei, ef, torch.cat([vf, mark], dim=1))
            out[r] = lg[r]
        return out

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        num_cons = constraint_features.shape[0]
        num_var = variable_features.shape[0]
        out = torch.zeros(num_var, device=variable_features.device,
                          dtype=variable_features.dtype)
        for c_ids, v_ids, lei in _graph_partition(edge_indices, num_cons, num_var, batch_indices):
            ef_g = _slice_edge_features(edge_features, edge_indices, batch_indices, v_ids)
            logits = self._run_one_graph(
                constraint_features[c_ids], lei, ef_g, variable_features[v_ids],
            )
            out[v_ids] = logits
        return out


def _slice_edge_features(edge_features, edge_index, batch_indices, v_ids_of_graph):
    """Select the edge-feature rows belonging to the graph whose variables are
    v_ids_of_graph (edges are ordered as in edge_index)."""
    if batch_indices is None or int(batch_indices.max()) == 0:
        return edge_features
    # membership mask by variable set of this graph
    vmask = torch.zeros(batch_indices.shape[0], dtype=torch.bool, device=edge_index.device)
    vmask[v_ids_of_graph] = True
    emask = vmask[edge_index[1]]
    return edge_features[emask]


# =========================================================================== #
#  6g. Edge-updating bipartite GNN (Graph-Network block; edge states EVOLVE)
# =========================================================================== #
class _EdgeUpdateLayer(nn.Module):
    """One full Graph-Network step (Battaglia et al. 2018; Gilmer et al. 2017):
    update every edge state from its incident constraint/variable states, then
    update each node type by aggregating its incident EDGE states. Node updates
    are type-specific (separate constraint / variable MLPs), so the bipartite
    structure is respected."""

    def __init__(self, emb_size):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, emb_size))
        self.cons_mlp = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, emb_size))
        self.var_mlp = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size), nn.ReLU(), nn.Linear(emb_size, emb_size))
        self.en = nn.LayerNorm(emb_size)
        self.cn = nn.LayerNorm(emb_size)
        self.vn = nn.LayerNorm(emb_size)

    def forward(self, h_c, h_v, h_e, edge_index):
        ci, vi = edge_index[0], edge_index[1]      # row0=constraint, row1=variable
        # (1) edge update: h_e <- f([h_e, h_cons(e), h_var(e)])
        h_e = self.en(h_e + self.edge_mlp(torch.cat([h_e, h_c[ci], h_v[vi]], dim=-1)))
        # (2) type-specific node updates by aggregating incident edge states
        agg_c = torch.zeros_like(h_c).index_add_(0, ci, h_e)
        agg_v = torch.zeros_like(h_v).index_add_(0, vi, h_e)
        h_c = self.cn(h_c + self.cons_mlp(torch.cat([h_c, agg_c], dim=-1)))
        h_v = self.vn(h_v + self.var_mlp(torch.cat([h_v, agg_v], dim=-1)))
        return h_c, h_v, h_e


class EdgeBipartiteGNNPolicy(nn.Module):
    """Edge-updating bipartite GNN. The default Gasse half-convolution consumes
    the coefficient a_ij as a STATIC message feature and never updates it; here a
    hidden state is maintained PER EDGE (per constraint-variable nonzero) and
    evolved every layer alongside the node states (see _EdgeUpdateLayer). The a_ij
    interaction is thus a first-class, evolving entity -- the "edge GNN" scheme
    used in several MILP solution-prediction / learning-to-branch works.

    Expressiveness note: this raises practical MODELING capacity for a_ij-
    dependent targets, but on the WL-separation axis it is still 1-WL-bounded --
    on edge-regular foldable graphs all edge states stay equal, so it does not
    separate them (richer edge modeling is orthogonal to beyond-1-WL separation;
    combine with rwse/substructure for that). Batching is exactly per-instance
    (all scatters use global indices on the block-diagonal graph)."""

    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
                 variable_nfeats=6, depth=4, jumping_knowledge=True):
        super().__init__()
        self.depth = depth
        self.jk = jumping_knowledge
        self.cons_embedding = nn.Sequential(
            nn.LayerNorm(constraint_nfeats), nn.Linear(constraint_nfeats, emb_size),
            nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU())
        self.var_embedding = nn.Sequential(
            nn.LayerNorm(variable_nfeats), nn.Linear(variable_nfeats, emb_size),
            nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU())
        self.edge_embedding = nn.Sequential(
            nn.LayerNorm(edge_nfeats), nn.Linear(edge_nfeats, emb_size),
            nn.ReLU(), nn.Linear(emb_size, emb_size), nn.ReLU())
        self.layers = nn.ModuleList([_EdgeUpdateLayer(emb_size) for _ in range(depth)])
        head_in = (depth + 1) * emb_size if jumping_knowledge else emb_size
        self.output_module = nn.Sequential(
            nn.Linear(head_in, emb_size), nn.ReLU(), nn.Linear(emb_size, 1, bias=False))

    def forward(self, constraint_features, edge_indices, edge_features,
                variable_features, batch_indices=None, is_training=False):
        h_c = self.cons_embedding(constraint_features)
        h_v = self.var_embedding(variable_features)
        h_e = self.edge_embedding(edge_features)
        v_hist = [h_v]
        for layer in self.layers:
            h_c, h_v, h_e = layer(h_c, h_v, h_e, edge_indices)
            v_hist.append(h_v)
        h = torch.cat(v_hist, dim=-1) if self.jk else h_v
        return self.output_module(h).squeeze(-1)


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
GNN_REGISTRY = {
    "gasse": GasseGNNPolicy,
    "bipartite_attention": BipartiteAttentionPolicy,
    "random_feature": RandomFeatureGNNPolicy,
    "tripartite": TripartiteGNNPolicy,
    "graph_transformer": GraphTransformerPolicy,
    # structural / positional / higher-expressivity encoders
    "bipartite_gin": BipartiteGINPolicy,
    "rwse": RWSEGNNPolicy,
    "substructure": SubstructureGNNPolicy,
    "graphgps": GraphGPSPolicy,
    "lap_pe": LapPEGNNPolicy,
    "id_gnn": IDGNNPolicy,
    "edge_gnn": EdgeBipartiteGNNPolicy,
}


def build_gnn(gnn_type, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
              variable_nfeats=6, **kwargs):
    """
    Construct a bipartite-tailored encoder by name. `gnn_type='gcn'` (the default
    2-layer GNNPolicy) and `'coco'`/`'moe'` are handled by their own modules and
    are intentionally not in this registry.
    """
    if gnn_type not in GNN_REGISTRY:
        raise ValueError(
            f"Unknown gnn_type '{gnn_type}'. Available here: {list(GNN_REGISTRY)} "
            f"(plus 'gcn'/'coco'/'moe' handled elsewhere)."
        )
    cls = GNN_REGISTRY[gnn_type]
    return cls(emb_size=emb_size, constraint_nfeats=constraint_nfeats,
               edge_nfeats=edge_nfeats, variable_nfeats=variable_nfeats, **kwargs)


# whitelist of gnn_types that share the plain PS forward signature and can be
# driven by PS_Family_Trainer + BCELossComputer and PSFamilyEvaluator's
# gcn-style predict branch.
PS_FAMILY_GNN_TYPES = ("gcn",) + tuple(GNN_REGISTRY)


def select_encoder_kwargs(gnn_type, cfg):
    """Collect the constructor kwargs relevant to `gnn_type` from a config
    mapping (OmegaConf DictConfig or plain dict -- both expose .get(key, default)).

    Single source of truth so that TRAIN-time and TEST-time model construction
    stay in sync; if they diverge, the checkpoint fails to load. Used by both
    scripts/train/train_gnn.py and scripts/test/test_ps.py.
    """
    g = cfg.get
    kw = {}
    deep = ("gasse", "bipartite_attention", "random_feature", "tripartite",
            "graph_transformer", "bipartite_gin", "rwse", "substructure",
            "graphgps", "lap_pe", "id_gnn", "edge_gnn")
    if gnn_type in deep:
        kw["depth"] = g("depth", 4)
        kw["jumping_knowledge"] = g("jumping_knowledge", True)
    if gnn_type in ("bipartite_attention", "graph_transformer", "graphgps"):
        kw["heads"] = g("heads", 4)
        kw["dropout"] = g("dropout", 0.0)
    if gnn_type == "random_feature":
        kw["n_rand"] = g("n_rand", 8)
    if gnn_type in ("rwse", "graphgps"):
        kw["walk_length"] = g("walk_length", 8)
    if gnn_type == "substructure":
        kw["powers"] = tuple(g("struct_powers", [4, 6]))
    if gnn_type == "bipartite_gin":
        kw["train_eps"] = g("train_eps", True)
    if gnn_type == "lap_pe":
        kw["pe_dim"] = g("pe_dim", 8)
    if gnn_type == "id_gnn":
        kw["max_roots"] = g("max_roots", 32)
    return kw


def build_ps_family_model(gnn_type, emb_size=64, constraint_nfeats=4,
                          edge_nfeats=2, variable_nfeats=6, **kwargs):
    """
    Factory for PS-family encoders that all expose the same forward signature
    (constraint_features, edge_indices, edge_features, variable_features,
    batch_indices, is_training) -> per-variable logits.

    Handles 'gcn' (the default GNNPolicy) plus the bipartite-tailored encoders
    in GNN_REGISTRY. `coco`/`moe` have bespoke signatures/trainers and are built
    by their own scripts.
    """
    if gnn_type == "gcn":
        from .gcn import GNNPolicy
        return GNNPolicy(emb_size=emb_size, constraint_nfeats=constraint_nfeats,
                         edge_nfeats=edge_nfeats, variable_nfeats=variable_nfeats)
    return build_gnn(gnn_type, emb_size=emb_size, constraint_nfeats=constraint_nfeats,
                     edge_nfeats=edge_nfeats, variable_nfeats=variable_nfeats, **kwargs)
