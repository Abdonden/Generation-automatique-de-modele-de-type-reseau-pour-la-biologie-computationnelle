import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool, global_max_pool

from embedding import MLP
from data import *




class Encoder (nn.Module):
    def __init__(self, n_species, d, dropout=0.1, num_layers = 1):
        super().__init__()
        self.h_0 = nn.Parameter(torch.randn(d)).reshape(1,1,d).repeat(num_layers, n_species, 1)
        self.mdl = nn.RNN(1, d, num_layers=num_layers, batch_first=True)

    def forward (self, x):
        n_points = x.size(0)

        return self.mdl(x, self.h_0)[0] # [n_species, n_timepoints, d]


class CausalGNN (nn.Module):
    def __init__(self, n_species, d, d_prime):
        super().__init__()
        self.A = nn.Parameter(torch.randn([n_species, n_species]))
        self.W = nn.Linear(d,d_prime)

    def forward (self, x):
        x = torch.matmul(self.A, x)
        x = self.W(x)
        return x


class CausalityInference (nn.Module):
    def __init__(self, n_species,
                 f_theta, # RNN encoder
                 g_mu, # decoder
                 d, d_prime # embedding dimensions
                 ):
        super().__init__()
        self.n_species = n_species
        self.encoder = f_theta
        self.decoder = g_mu
        self.gnn = CausalGNN(n_species, d, d_prime)

    def forward(self,
                x): #[n_species, n_timepoints]
        x = x.unsqueeze(-1)
        x = self.encoder(x) # [n_species, n_timepoints, d] : embeddings des timepoints de chaque serie
        x = x.transpose(0,1) # [n_timepoints,  n_species, d] : feature matrix at each timepoint
        x = self.gnn(x) # [n_timepoints, n_species,d']
        x = self.decoder(x) # [n_timepoints, n_species, 1]

        x = x.transpose(0,1) # [n_species, n_timepoints]
        return x.squeeze()





series = torch.rand([5, 10]) # 5 species, 10 timepoints
encoder = Encoder(5, 2) # 5 species, d=2
decoder = nn.Linear(3,1) # d'=3
c_mdl = CausalityInference(5, encoder, decoder, 2, 3)




