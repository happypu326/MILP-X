import torch
import torch_geometric
import torch.nn as nn
from torch_geometric.nn import GraphNorm


class CoCoGNNPolicy(torch.nn.Module):
    def __init__(self, emb_size, cons_nfeats, edge_nfeats, var_nfeats, depth=2, Intra_Constraint_Competitive=True):
        super().__init__()
        self.embd_size = emb_size
        self.cons_nfeats = cons_nfeats
        self.edge_nfeats = edge_nfeats
        self.var_nfeats = var_nfeats
        self.depth = depth

        self.cons_embedding = nn.Sequential(
            nn.BatchNorm1d(self.cons_nfeats),
            nn.Linear(self.cons_nfeats, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
            nn.ReLU(),
        )

        self.edge_embedding = nn.Sequential(
            nn.BatchNorm1d(self.edge_nfeats),
        )

        self.var_embedding = nn.Sequential(
            nn.BatchNorm1d(self.var_nfeats),
            nn.Linear(self.var_nfeats, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, self.embd_size),
            nn.ReLU(),
        )

        conv_v_to_c_list = []
        graph_norm_v_to_c_list = []
        conv_c_to_v_list = []
        graph_norm_c_to_v_list = []

        for _ in range(self.depth):
            conv_v_to_c_list.append(BipartiteGraphConvolution())
            graph_norm_v_to_c_list.append(GraphNorm(self.embd_size))
            conv_c_to_v_list.append(BipartiteGraphConvolution())
            graph_norm_c_to_v_list.append(GraphNorm(self.embd_size))

        # Now set the Sequential containers
        self.conv_v_to_c_layers = nn.Sequential(*conv_v_to_c_list)
        self.graph_norm_v_to_c_layers = nn.Sequential(*graph_norm_v_to_c_list)
        self.conv_c_to_v_layers = nn.Sequential(*conv_c_to_v_list)
        self.graph_norm_c_to_v_layers = nn.Sequential(*graph_norm_c_to_v_list)


        self.output_norm = GraphNorm((self.depth + 1) * self.embd_size)
        self.vars_output_layer = torch.nn.Sequential(
            nn.Linear((self.depth + 1) * self.embd_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )
        
        self.cons_output_layer = torch.nn.Sequential(
            nn.Linear((self.depth + 1) * self.embd_size, self.embd_size),
            nn.ReLU(),
            nn.Linear(self.embd_size, 1) 
        )

        self.Intra_Constraint_Competitive = Intra_Constraint_Competitive
        if Intra_Constraint_Competitive:
            constraint_normalization_layers = []
            for _ in range(self.depth):
                constraint_normalization_layers.append(ConstraintNormalization())
            self.constraint_normalization_layers = nn.Sequential(*constraint_normalization_layers)

    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features, n_constraints=None, constraint_features_batch=None, variable_features_batch = None
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension (64)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        vars_outputs = [variable_features]
        cons_outputs = [constraint_features]
        for i in range(self.depth):            
            constraint_features = self.conv_v_to_c_layers[i](
                variable_features, reversed_edge_indices, edge_features, constraint_features
            ) #+ constraint_features
            variable_features = self.conv_c_to_v_layers[i](
                constraint_features, edge_indices, edge_features, variable_features
            ) #+ variable_features

            if self.Intra_Constraint_Competitive:
                variable_features = self.constraint_normalization_layers[i](
                    variable_features, reversed_edge_indices, edge_features, constraint_features, n_constraints
                )
            
            constraint_features = self.graph_norm_v_to_c_layers[i](
                constraint_features, constraint_features_batch)
            variable_features = self.graph_norm_c_to_v_layers[i](
                variable_features, variable_features_batch)
            
            vars_outputs.append(variable_features)
            cons_outputs.append(constraint_features)

        variable_features = torch.cat(vars_outputs, dim=-1)
        vars_out = self.vars_output_layer(variable_features).squeeze(-1)

        return vars_out

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("add")
        emb_size = 64

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
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
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output


class ConstraintNormalization(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self):
        super().__init__("mean")
        self.gamma = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, variable_features, edge_indices, edge_features, constraint_features, n_constraints=None):
        # Remove the last constraint node which connects to all variables in each graph in batch
        if n_constraints is not None and isinstance(n_constraints, torch.Tensor):
            # Create a mask to filter out edges connected to the last constraint node of each graph
            mask = torch.ones(edge_indices.shape[1], dtype=torch.bool, device=edge_indices.device)
            
            # Calculate the indices of the last constraint nodes for all graphs at once
            last_constraint_indices = torch.cumsum(n_constraints, dim=0) - 1
            
            # Create the mask in a vectorized way
            for last_idx in last_constraint_indices:
                mask &= (edge_indices[1] != last_idx)
            
            # Apply the mask to filter edges
            filtered_edge_indices = edge_indices[:, mask]
            filtered_edge_features = edge_features[mask]
        else:
            assert n_constraints is not None
            max_constraint_idx = edge_indices[1].max()
            assert n_constraints == max_constraint_idx + 1
            assert sum(edge_indices[1] == max_constraint_idx) == len(variable_features)
            mask = edge_indices[1] != max_constraint_idx
            filtered_edge_indices = edge_indices[:, mask]
            filtered_edge_features = edge_features[mask]
        # Propagate using the filtered edges
        right_avg = self.propagate(
            filtered_edge_indices,
            size=(variable_features.shape[0], constraint_features.shape[0]),
            node_features=(variable_features, constraint_features),
            edge_features=filtered_edge_features,
        )
        
        # Reverse the filtered edges for the second propagation
        reversed_edge_indices = torch.stack([filtered_edge_indices[1], filtered_edge_indices[0]], dim=0)
        left_avg = self.propagate(
            reversed_edge_indices,
            size=(constraint_features.shape[0], variable_features.shape[0]),
            node_features=(right_avg, variable_features),
            edge_features=filtered_edge_features,
        )
        
        normalized_features = variable_features - self.beta * left_avg
        # normalized_features = (variable_features - self.beta * left_avg) / self.gamma
        return normalized_features
    
    def message(self, node_features_i, node_features_j, edge_features):
        return node_features_j