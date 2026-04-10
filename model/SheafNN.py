import torch.nn as nn
import torch.nn.functional as F
import torch
from model.gnns import MLP
from util.laplacian_builder import GeneralLaplacianBuilder
import torch
import torch_sparse


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
                 mlp_hidden_channels: list = [64, 64, 64], stalk: int = 2, act: str = 'F.elu'):
        super().__init__()
        assert hidden_channels % stalk == 0, "Hidden channels must be divisible by the stalk dimension"
        self.hidden_channels = hidden_channels
        self.stalk = stalk
        self.dropout = dropout
        self.n_layers = n_layers  
        self.maps_bool = MLP_maps

        self.act = eval(act)
        self.maps = None  # Will be initialized in the forward pass
        self.laplacian_builder = None  # Will be initialized in the forward pass

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

        # self.print = True  # Debug flag to print shapes only once
    def _init_maps(self, edge_index, device):
        # We expect edge_index to be directional
        num_edges = edge_index.size(1)
        self.maps = nn.Parameter(
            torch.randn(num_edges, self.stalk, self.stalk, device=device)
        )

    def _init_MLP_maps(self, edge_index, x):
        # The function expects directional edge_index
        num_edges = edge_index.size(1)
        source, destination = edge_index[0,:], edge_index[1,:]

        embed = torch.cat((x[source], x[destination]), dim = 1)

        maps = self.MLP_maps(embed)
        # We try to avoid laplacian explosion by limiting maps
        # maps = torch.tanh(maps) *0.1
        self.maps = maps.reshape((num_edges, self.stalk, self.stalk))

    def _forward_body(self, data):

        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp_in(x)

        if self.laplacian_builder is None:
            # The normalizatin is the exact same one of the Sheaf Diffusion paper, but it might be instable
            self.laplacian_builder = GeneralLaplacianBuilder(size = N, edge_index = edge_index, d = self.stalk, normalised = True)

        if self.maps is None and not self.maps_bool:
             self._init_maps(edge_index, x.device)
        elif self.maps_bool:
             self._init_MLP_maps(edge_index, x)

        x = x.reshape((N * self.stalk, -1))
        laplacian, _ = self.laplacian_builder(self.maps)
        index, value = laplacian
        
        for layer in range(self.n_layers):
            W1 = self.linear_layers[2 * layer]
            W2 = self.linear_layers[2 * layer + 1]

            x_W2 = x @ W2.weight.T.contiguous()
            f = x_W2.size(-1)
            x_reshaped = x_W2.reshape((N, self.stalk, f))

            H_tensor = torch.einsum('ij, njk -> nik', W1.weight.T.contiguous(), x_reshaped)
            H = H_tensor.reshape(N * self.stalk, -1)

            H = torch_sparse.spmm(index, value, x.size(0), x.size(0), H)
            x = x - self.act(H)

            if layer < self.n_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        """
            if self.training and self.print:
                print(f"[DEBUG] Stalk configurato: {self.stalk}, hidden_channels: {self.hidden_channels}")
                print(f"[DEBUG] Shape self.maps: {self.maps.shape} | Atteso: [E, {self.stalk}, {self.stalk}]")
                print(f"[DEBUG] Shape W1: {self.linear_layers[0].weight.shape}")
                print(f"[DEBUG] MLP_maps: {getattr(self, 'MLP_maps', None)}, expected MLP_maps: {self.maps_bool}")
                self.print = False
        """
        return x.reshape((-1, self.hidden_channels))
    
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

            self.laplacian_builder = None  # Clear laplacian builder to ensure it is re-initialized on the next forward pass