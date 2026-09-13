import torch
import torch.nn.functional as F
import torch_geometric


class GNNPolicy(torch.nn.Module):
    def __init__(self, emb_size=64, constraint_nfeats=4, edge_nfeats=1, variable_nfeats=6):
        super().__init__()
        self.encoder = GNNEncoder(emb_size, constraint_nfeats, edge_nfeats, variable_nfeats)
        self.decoder = GNNDecoder(emb_size)

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        batch_indices=None,
        is_training=False,
    ):
        encoder_out = self.encoder(
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,
            batch_indices=batch_indices,
        )
        if len(encoder_out) == 2:
            variable_features, constraint_features = encoder_out
            cbloss = None
        else:
            variable_features, constraint_features, cbloss = encoder_out
        output = self.decoder(variable_features)
        return output


class GNNEncoder(torch.nn.Module):
    def __init__(
        self,
        emb_size=64,
        constraint_nfeats=4,
        edge_nfeats=1,
        variable_nfeats=6,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.cons_nfeats = constraint_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = variable_nfeats

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.cons_nfeats),
            torch.nn.Linear(self.cons_nfeats, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(self.var_nfeats),
            torch.nn.Linear(self.var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, edge_nfeats=self.edge_nfeats)
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, edge_nfeats=self.edge_nfeats)

        self.conv_v_to_c2 = BipartiteGraphConvolution(self.emb_size, edge_nfeats=self.edge_nfeats)
        self.conv_c_to_v2 = BipartiteGraphConvolution(self.emb_size, edge_nfeats=self.edge_nfeats)

    def forward(
        self,
        constraint_features,
        edge_indices,
        edge_features,
        variable_features,
        batch_indices=None,
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )

        constraint_features = self.conv_v_to_c2(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v2(
            constraint_features, edge_indices, edge_features, variable_features
        )

        return variable_features, constraint_features


class GNNDecoder(torch.nn.Module):
    def __init__(self, emb_size=64):
        super().__init__()
        self.emb_size = emb_size
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, 1, bias=False),
        )

    def forward(self, variable_features):
        # A final MLP on the variable features
        output = self.output_module(variable_features).squeeze(-1)
        return output


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self, emb_size=64, edge_nfeats=1):
        super().__init__("add")
        self.emb_size = emb_size
        self.edge_nfeats = edge_nfeats

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(self.edge_nfeats, self.emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(self.emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        # node_features_i,the node to be aggregated
        # node_features_j,the neighbors of the node i

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output
