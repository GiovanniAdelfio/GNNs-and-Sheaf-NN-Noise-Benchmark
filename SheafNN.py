import torch.nn as nn
import torch.nn.functional as F
import torch
from gnns import MLP
import itertools
import torch
import torch_sparse

from torch_geometric.utils import degree
# Forse da sostituire con coalesce from torch_geometric.utils, da verificare se è equivalente
def remove_duplicate_edges(edge_index):
    processed_edges = set()
    new_edge_index = []

    for e in range(edge_index.size(1)):
        source, target = sorted((edge_index[0, e].item(), edge_index[1, e].item()))
        if (source, target) in processed_edges:
            continue
        processed_edges.add((source, target))
        new_edge_index.append([source, target])
    print(f"Removed {edge_index.size(1) - len(new_edge_index)} edges")
    return torch.tensor(new_edge_index, dtype=torch.long).t()

# BUG: torch_sparse.spspmm does NOT support autograd from 0.44 forward, needs a new implementation or using torch module
def build_sheaf_laplacian(N, K, edge_index, maps):
    """
    Builds a sheaf laplacian given the edge_index and the restriction maps

    Args:
        N: The number of nodes in the graph
        K: The dimensionality of the Stalks
        edge_index: Edge index of the graph without duplicate edges. We assume that edge i has orientation
            edge_index[0, i] --> edge_index[1, i].
        maps: Tensor of shape [edge_index.size(1), 2 (source/target), K, K] containing the restriction maps of the sheaf
    Returns:
        (index, value): The sheaf Laplacian as a sparse matrix of size (N*K, N*K)
    """
    E = edge_index.size(1)
    index = []
    values = []

    for e in range(E):
        source = edge_index[0, e]
        target = edge_index[1, e]

        top_x = e * K
        # Generate the positions in the block matrix
        top_y = source * K
        for i, j in itertools.product(range(K), range(K)):
            index.append([top_x + i, top_y + j])
            values.append(-maps[e, 0, i, j])

        top_y = target * K
        for i, j in itertools.product(range(K), range(K)):
            index.append([top_x + i, top_y + j])
            values.append(maps[e, 1, i, j])

    index = torch.tensor(index, dtype=torch.long).T
    #values = torch.tensor(values)           original bodnar version, but it blocks gradient flow 
    values = torch.stack(values)           # this version allows gradient flow through the maps

    index_t, values_t = torch_sparse.transpose(index, values, E * K, N * K)
    index, value = torch_sparse.spspmm(index_t, values_t, index, values, N * K, E * K, N * K, coalesced=True)
    return torch_sparse.coalesce(index, value, N * K, N * K)


class SheafNN(nn.Module):
    """
    A simple implementation of Sheaf Neural Networks (SheafNN) for node classification tasks.
    1. It has a first MLP which makes an embedding of the input features, dim in_channels --> hidden_channels which must be divisible by stalk;
    2. initialization of the restriction maps/MLP for generation of the restriction maps on the first forward;
    3. generate the Laplacian and apply diffusion for n_layer times, implemented as in Sheaf Diffusion by Bodnar;
    4. finally I apply the MLP out to generate the probability vectors form the final embeddings, dim hidden_channels --> out_channels;
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int = 2, 
                 dropout: float = 0.5, mlp_layers: list = [2, 2, 2], MLP_maps: bool = True,  
                 mlp_hidden_channels: list = [64, 64, 64], stalk: int = 2, act: str = 'F.relu'):
        super().__init__()
        assert hidden_channels % stalk == 0, "Hidden channels must be divisible by the stalk dimension"
        self.hidden_channels = hidden_channels
        self.stalk = stalk
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.mlp_hidden_channels = mlp_hidden_channels
        self.mlp_layers = mlp_layers
        self.maps_bool = MLP_maps

        # In caso sostituisco con eval di torch.nn.functional
        if act == 'F.relu':
            self.act = F.relu
        elif act == 'F.leaky_relu':
            self.act = F.leaky_relu
        elif act == "F.sigmoid":
            self.act = F.sigmoid
        elif act == "F.tanh":
            self.act = F.tanh
        else:
            raise ValueError(f"Unsupported activation function: {act}")

        self.maps = None  # Will be initialized in the forward pass
        self.cached_edge_index = None  # Cache for edge_index without duplicates

        self.mlp_in = MLP(in_channels, mlp_hidden_channels[0], hidden_channels, 
                       mlp_layers[0], dropout)
        self.mlp_out = MLP(hidden_channels, mlp_hidden_channels[1], out_channels, 
                        mlp_layers[1], dropout)
        if self.maps_bool:
            assert len(mlp_layers) == 3, "Expected 3 values in mlp_layers when MLP_maps is True"
            self.MLP_maps = MLP(hidden_channels *2, mlp_hidden_channels[2], stalk ** 2, 
                                mlp_layers[2], dropout = 0)
        
        self.linear_layers = nn.ModuleList()
        f = hidden_channels // stalk
        for _ in range(n_layers):
            self.linear_layers.append(nn.Linear(stalk, stalk, bias=False))
            self.linear_layers.append(nn.Linear(f, f, bias=False))


    def _init_maps(self, edge_index, device):
        num_edges = edge_index.size(1)
        self.maps = nn.Parameter(
            torch.randn(num_edges, 2, self.stalk, self.stalk, device=device)
        )

    def _init_MLP_maps(self, edge_index, x):
        num_edges = edge_index.size(1)
        source, destination = edge_index[0,:], edge_index[1,:]

        embed_1 = torch.cat((x[source], x[destination]), dim= 1)
        embed_2 = torch.cat((x[destination], x[source]), dim= 1)

        maps_1 = self.MLP_maps(embed_1)
        maps_2 = self.MLP_maps(embed_2)

        maps = torch.stack((maps_1, maps_2), dim=1)
        self.maps = maps.view( num_edges, 2, self.stalk, self.stalk)

    def _forward_body(self, data):
        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp_in(x)

        if self.cached_edge_index is None:
            self.cached_edge_index = remove_duplicate_edges(edge_index)

        if self.maps is None and not self.maps_bool:
             self._init_maps(self.cached_edge_index, x.device)
        elif self.maps_bool:
             self._init_MLP_maps(self.cached_edge_index, x)

        x = x.view(N * self.stalk, -1)
        In = torch.eye(N, device=x.device)
        laplacian = build_sheaf_laplacian(N, self.stalk, self.cached_edge_index, self.maps)
        index, value = laplacian
        
        for layer in range(self.n_layers):
            W1 = self.linear_layers[2 * layer]
            W2 = self.linear_layers[2 * layer + 1]
            
            H = torch.kron(In, W1.weight.T) @ x @ W2.weight.T
            H = torch_sparse.spmm(index, value, x.size(0), x.size(0), H)
            x = x - self.act(H)
            if layer < self.n_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.view(-1, self.hidden_channels)
    
    def get_embeddings(self, data):
        """Return raw hidden representation before the final projection and its transforms."""
        return self._forward_body(data)

    def forward(self, data):
        x = self._forward_body(data)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp_out(x)
    
    def initialize(self):

            for module in self.mlp_in.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

            for module in self.mlp_out.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

            for layer in self.linear_layers:
                layer.reset_parameters()
                
            if self.maps is not None and not self.maps_bool:
                nn.init.normal_(self.maps)
            elif self.maps_bool:
                for module in self.MLP_maps.modules():
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
            self.cached_edge_index = None  # Clear cached edge index to ensure it is recomputed if needed