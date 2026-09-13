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
    shipped GNNPolicy but with configurable depth and jumping-knowledge (concat
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


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
GNN_REGISTRY = {
    "gasse": GasseGNNPolicy,
    "bipartite_attention": BipartiteAttentionPolicy,
    "random_feature": RandomFeatureGNNPolicy,
    "tripartite": TripartiteGNNPolicy,
    "graph_transformer": GraphTransformerPolicy,
}


def build_gnn(gnn_type, emb_size=64, constraint_nfeats=4, edge_nfeats=2,
              variable_nfeats=6, **kwargs):
    """
    Construct a bipartite-tailored encoder by name. `gnn_type='gcn'` (the shipped
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


def build_ps_family_model(gnn_type, emb_size=64, constraint_nfeats=4,
                          edge_nfeats=2, variable_nfeats=6, **kwargs):
    """
    Factory for PS-family encoders that all expose the same forward signature
    (constraint_features, edge_indices, edge_features, variable_features,
    batch_indices, is_training) -> per-variable logits.

    Handles 'gcn' (the shipped GNNPolicy) plus the bipartite-tailored encoders
    in GNN_REGISTRY. `coco`/`moe` have bespoke signatures/trainers and are built
    by their own scripts.
    """
    if gnn_type == "gcn":
        from .gcn import GNNPolicy
        return GNNPolicy(emb_size=emb_size, constraint_nfeats=constraint_nfeats,
                         edge_nfeats=edge_nfeats, variable_nfeats=variable_nfeats)
    return build_gnn(gnn_type, emb_size=emb_size, constraint_nfeats=constraint_nfeats,
                     edge_nfeats=edge_nfeats, variable_nfeats=variable_nfeats, **kwargs)
