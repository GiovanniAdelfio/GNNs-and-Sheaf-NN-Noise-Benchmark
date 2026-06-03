import torch.nn as nn
import torch.nn.functional as F
import torch
from util.laplacian_builder import GeneralLaplacianBuilder
import torch
import torch_sparse
from torch_householder import torch_householder_orgqr


class SheafNN_Light(nn.Module):
    """
    An implementation which follows directly the orthogonal maps implementation of the Sheaf by Bodnar.
    1. It has a first linear layer+ activation which makes an embedding of the input features, dim in_channels --> hidden_channels which must be divisible by stalk;
    2. initialization of the linear layer for generation of the restriction maps on the forward based on the concatenation of the nodes' features;
    3. generate the Laplacian and apply diffusion for n_layer times;
    4. finally I apply the linear layer out to generate the probability vectors form the final embeddings, dim hidden_channels --> out_channels;
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 n_layers: int = 2, 
                 dropout_in: float = 0.5,
                 dropout: float = 0.5, 
                 stalk: int = 2, 
                 act: str = 'F.elu',
                 norm_info: str = None):
        
        super().__init__()
        assert hidden_channels % stalk == 0, "Hidden channels must be divisible by the stalk dimension"
        self.hidden_channels = hidden_channels
        self.stalk = stalk
        self.dropout_in = dropout_in  
        self.dropout = dropout
        self.n_layers = n_layers  

        self.act = eval(act)
        self.laplacian_builder = None  # Will be initialized in the forward pass

        self.emb_in = nn.Linear(in_channels, hidden_channels)
        self.emb_out = nn.Linear(hidden_channels, out_channels)

        self.gen_maps = nn.Linear(2 * hidden_channels, stalk**2, bias=False)

        self.linear_layers = nn.ModuleList()
        f = hidden_channels // stalk
        self.linear_layers.append(nn.Linear(stalk, stalk, bias=False))
        self.linear_layers.append(nn.Linear(f, f, bias=False))

        for layer in self.linear_layers:
            nn.init.eye_(layer.weight)
            
        self.step_size = torch.nn.Parameter(torch.tensor(1.0))  # Learnable step size for diffusion

    def _init_maps(self, edge_index, x):
            """
            Genera mappe di restrizione ORTOGONALI usando le riflessioni di Householder.
            x: tensore degli embedding [N, hidden_channels] (output di mlp_in)
            """
            num_edges = edge_index.size(1)
            source, destination = edge_index[0, :], edge_index[1, :]

            embed = torch.cat((x[source], x[destination]), dim=1)

            params = self.gen_maps(embed).reshape(num_edges, self.stalk, self.stalk)
            params = F.tanh(params)  
            
            # Inizializziamo le mappe finali come matrici Identità 
            eye = torch.eye(self.stalk, device=x.device).unsqueeze(0).repeat(num_edges, 1, 1)
            A = params.tril(diagonal=-1) + eye
            
            # Chiamata alla libreria C++ ottimizzata
            self.maps = torch_householder_orgqr(A)

    def _diffusion(self, x, N, laplacian):
        index, value = laplacian

        W1 = self.linear_layers[0]
        W2 = self.linear_layers[1]
        
        for layer in range(self.n_layers):
            x_W2 = x @ W2.weight.T.contiguous()
            f = x_W2.size(-1)
            x_reshaped = x_W2.reshape((N, self.stalk, f))

            H_tensor = torch.einsum('ij, njk -> nik', W1.weight.T.contiguous(), x_reshaped)
            H = H_tensor.reshape(N * self.stalk, -1)

            H = torch_sparse.spmm(index, value, x.size(0), x.size(0), H)
            x = x - self.step_size * self.act(H)

            if layer < self.n_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def _forward_body(self, data):

        x, edge_index = data.x, data.edge_index
        N = x.size(0)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = self.emb_in(x)
        x = self.act(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.laplacian_builder is None:
            # The normalizatin is the exact same one of the Sheaf Diffusion paper, but it might be instable
            self.laplacian_builder = GeneralLaplacianBuilder(size = N, edge_index = edge_index, d = self.stalk, normalised = False, deg_normalised = True)

        self._init_maps(edge_index, x)
        x = x.reshape((N * self.stalk, -1))
        laplacian, _ = self.laplacian_builder(self.maps) 
        
        # Implementation of the ego method from Zhu et al 2020
        x = self._diffusion(x, N, laplacian)

        return x.reshape(-1, self.hidden_channels)
    
    def get_embeddings(self, data):
        """Return raw hidden representation before the final projection and its transforms."""
        return self._forward_body(data)

    def forward(self, data):
        x = self._forward_body(data)
        x = self.emb_out(x)
        return x

    def initialize(self):

            for module in self.emb_in.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

            for module in self.emb_out.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

            for layer in self.linear_layers:
                nn.init.eye_(layer.weight)
                
            self.gen_maps.reset_parameters()

            self.laplacian_builder = None  # Clear laplacian builder to ensure it is re-initialized on the next forward pass