import pickle
import os
import torch
import torch_geometric
import numpy as np
from src.utils.utils import build_edge_features

class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files, method_type, use_edge_coeff=True):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.method_type = method_type
        # When True, edges carry [normalized a_ij, sign(a_ij)] (edge_nfeats=2).
        # When False, reproduce the original PS behaviour: every edge = 1
        # (edge_nfeats=1). See src.utils.utils.build_edge_features.
        self.use_edge_coeff = use_edge_coeff

    def len(self):
        return len(self.sample_files)

    def process_sample(self,filepath):
        if self.method_type == 'DiffILO':
            BGFilepath = filepath
            TensorFilepath = filepath.replace("samples", "tensors")

            with open(BGFilepath, "rb") as f:
                graph = pickle.load(f)
                
            with open(TensorFilepath, "rb") as f:
                Tensors = pickle.load(f)
        
            return graph, Tensors
        else:
            BGFilepath, solFilePath, group = filepath
            with open(BGFilepath, "rb") as f:
                bgData = pickle.load(f)
            with open(solFilePath, "rb") as f:
                solData = pickle.load(f)

            BG = bgData
            varNames = solData['var_names']

            sols = solData['sols'][:50]#[0:300]
            objs = solData['objs'][:50]#[0:300]

            sols=np.round(sols,0)
            return BG, sols, objs, varNames, group

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        if self.method_type == 'DiffILO':
            BGFilePath = self.sample_files[index]
            TensorFilepath = BGFilePath.replace("samples", "tensors")

            with open(BGFilePath, "rb") as f:
                BG = pickle.load(f)
                
            with open(TensorFilepath, "rb") as f:
                Tensors = pickle.load(f)

            constraint_features, edge_indices, edge_features, variable_features = BG

            graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features)
            graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
            
            graph.A = Tensors[0]
            graph.b = Tensors[1]
            graph.c = Tensors[2]

            return graph
        elif self.method_type == 'EnCore':
            return self._get_encore(index)
        elif self.method_type == 'CTC':
            return self._get_ctc(index)
        elif self.method_type == 'CLLNS':
            return self._get_cllns(index)
        else:
            BG, sols, objs, varNames, group = self.process_sample(self.sample_files[index])

            A, v_map, v_nodes, c_nodes, b_vars=BG

            constraint_features = c_nodes
            edge_indices = A._indices()

            variable_features = v_nodes
            # Edge features: real coefficient a_ij (normalized per-constraint +
            # sign) when use_edge_coeff, else the legacy constant-1 edge.
            edge_values = A._values()
            edge_features = build_edge_features(
                edge_indices,
                edge_values,
                use_edge_coeff=self.use_edge_coeff,
                num_cons=constraint_features.shape[0],
            )

            constraint_features[torch.isnan(constraint_features)] = 1

            graph = BipartiteNodeData(
                torch.FloatTensor(constraint_features.cpu()),
                torch.LongTensor(edge_indices.cpu()),
                torch.FloatTensor(edge_features.cpu()),
                torch.FloatTensor(variable_features.cpu())
            )
            
            graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
            graph.solutions = torch.FloatTensor(sols).reshape(-1)

            graph.objVals = torch.FloatTensor(objs)
            graph.nsols = sols.shape[0]
            graph.ntvars = variable_features.shape[0]
            graph.ntcons = constraint_features.shape[0]
            graph.varNames = varNames
            varname_dict={}
            varname_map=[]
            i=0
            for iter in varNames:
                varname_dict[iter]=i
                i+=1
            for iter in v_map:
                varname_map.append(varname_dict[iter])

            varname_map=torch.tensor(varname_map)

            graph.varInds = [[varname_map],[b_vars]]
            graph.n_constraints = constraint_features.shape[0]
            graph.group = group

            return graph

    def _get_encore(self, index):
        """EnCore sample: bipartite graph + the early solution x_ES appended as a
        7th variable feature, and per-variable early-to-final consistency labels
        y_i = 1[round(x*_i) == round(x_ES_i)] (x* = best full-budget solution)."""
        BGFilepath, solFilePath, group = self.sample_files[index]
        with open(BGFilepath, "rb") as f:
            BG = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        A, v_map, v_nodes, c_nodes, b_vars = BG
        varNames = solData['var_names']
        sols = np.round(solData['sols'][:50], 0)
        early_sols = solData.get('early_sols', [])

        varname_dict = {n: i for i, n in enumerate(varNames)}
        varname_map = [varname_dict[n] for n in v_map]        # graph idx -> var_names idx
        vm = np.array(varname_map)

        n_var = len(v_map)
        x_star = sols[0][vm] if len(sols) > 0 else np.zeros(n_var, dtype=np.float32)
        if len(early_sols) > 0:
            # train on a randomly chosen early solution, matching the inference-time
            # ensemble over the K probe incumbents (avoids over-fitting to only the best)
            xk = np.asarray(early_sols[np.random.randint(len(early_sols))])
            x_es = np.round(xk)[vm]
        elif len(sols) > 1:
            # no probe incumbent: use the worst pooled solution as a proxy early
            # incumbent. NOT x_star -- feeding the final solution would leak the
            # target and make every consistency label 1.
            x_es = np.round(sols[-1])[vm]
        else:
            x_es = x_star.copy()          # single solution only: degenerate fallback
        consistency = (np.round(x_star) == np.round(x_es)).astype(np.float32)

        constraint_features = c_nodes
        constraint_features[torch.isnan(constraint_features)] = 1
        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices, A._values(), use_edge_coeff=self.use_edge_coeff,
            num_cons=constraint_features.shape[0])

        variable_features = v_nodes.cpu()
        early_col = torch.FloatTensor(x_es).reshape(-1, 1)
        variable_features = torch.cat([variable_features, early_col], dim=1)  # [n_var, 7]

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            variable_features,
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.consistency_labels = torch.FloatTensor(consistency)
        graph.nsols = 1
        graph.ntvars = variable_features.shape[0]
        graph.ntcons = constraint_features.shape[0]
        graph.varNames = varNames
        graph.varInds = [[torch.tensor(varname_map)], [b_vars]]
        graph.group = group
        return graph

    def _get_ctc(self, index):
        """Constraint Matters sample: bipartite graph + per-variable assignment
        labels (best solution) + per-constraint critical-tight-constraint labels
        (aligned to constraint nodes; the appended objective node gets label 0)."""
        BGFilepath, solFilePath, group = self.sample_files[index]
        with open(BGFilepath, "rb") as f:
            BG = pickle.load(f)
        with open(solFilePath, "rb") as f:
            solData = pickle.load(f)

        A, v_map, v_nodes, c_nodes, b_vars = BG
        varNames = solData['var_names']
        sols = np.round(solData['sols'][:50], 0)
        ctc = np.asarray(solData.get('ctc_labels', []), dtype=np.float32)

        varname_dict = {n: i for i, n in enumerate(varNames)}
        varname_map = [varname_dict[n] for n in v_map]
        vm = np.array(varname_map)
        n_var = len(v_map)
        x_star = sols[0][vm] if len(sols) > 0 else np.zeros(n_var, dtype=np.float32)

        constraint_features = c_nodes
        constraint_features[torch.isnan(constraint_features)] = 1
        ncons_total = constraint_features.shape[0]           # #real constraints + objective node
        ctc_full = np.zeros(ncons_total, dtype=np.float32)   # objective node (last row) stays 0
        m = min(len(ctc), ncons_total - 1)
        ctc_full[:m] = ctc[:m]

        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices, A._values(), use_edge_coeff=self.use_edge_coeff,
            num_cons=ncons_total)

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            torch.FloatTensor(v_nodes.cpu()),
        )
        graph.num_nodes = ncons_total + n_var
        graph.var_labels = torch.FloatTensor(x_star)
        graph.ctc_labels = torch.FloatTensor(ctc_full)
        graph.nsols = 1
        graph.ntvars = n_var
        graph.ntcons = ncons_total
        graph.varNames = varNames
        graph.varInds = [[torch.tensor(varname_map)], [b_vars]]
        graph.group = group
        return graph

    def _get_cllns(self, index):
        """CL-LNS state: bipartite graph with a W-step incumbent-value history
        appended to the variable features, plus positive / negative destroy
        actions over the binary variables. Use with batch_size=1."""
        path = self.sample_files[index]
        with open(path, "rb") as f:
            st = pickle.load(f)

        A, v_map, v_nodes, c_nodes, b_vars = st['bg']
        window = torch.FloatTensor(np.asarray(st['window'], dtype=np.float32))   # [n_var, W]

        constraint_features = c_nodes
        constraint_features[torch.isnan(constraint_features)] = 1
        edge_indices = A._indices()
        edge_features = build_edge_features(
            edge_indices, A._values(), use_edge_coeff=self.use_edge_coeff,
            num_cons=constraint_features.shape[0])
        variable_features = torch.cat([v_nodes.cpu(), window], dim=1)            # [n_var, 6+W]

        graph = BipartiteNodeData(
            torch.FloatTensor(constraint_features.cpu()),
            torch.LongTensor(edge_indices.cpu()),
            torch.FloatTensor(edge_features.cpu()),
            variable_features,
        )
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.pos_actions = torch.FloatTensor(np.asarray(st['pos_actions'], dtype=np.float32))
        graph.neg_actions = torch.FloatTensor(np.asarray(st['neg_actions'], dtype=np.float32))
        graph.nsols = 1
        graph.ntvars = variable_features.shape[0]
        graph.varInds = [[torch.arange(len(v_map))], [b_vars]]
        graph.group = 0
        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features = None,
            edge_indices = None,
            edge_features = None,
            variable_features = None

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features

    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates":
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
