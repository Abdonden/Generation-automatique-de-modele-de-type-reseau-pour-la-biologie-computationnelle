import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import MLP
from data import *

# Idée: 
#   1. En utilisant le même modele: projeter chaque série temporelle dans un espace latent H = f(X)
#   2. Ajuster les coefficient d'une matrice A
#   3. Caclculer X' = AH + eps avec eps un bruit blanc utilisant le *reparameterisation trick*
#   4. Minimiser ||X' - X||

# Notes, on pourrait appliquer l'étape 3 plusieurs fois (comme plusierus couches de GNN avec une matrice apprise)


class TemporalEncoder (nn.Module):
    """
        Step 1: encodes the time series
    """
    def __init__ (self, n_points, n_species, hidden_size, dropout=0.2):
        super().__init__()
        self.n_points, self.n_species = n_points, n_species
        self.mlp = MLP(n_points, n_species,hidden_size, dropout=dropout)

    def forward(self, x):
        x = self.mlp(x)
        return x



class GraphGenerator (nn.Module):
    """
        Step 2 : constructs an adjacency matrix using the temporal series

    """

    def __init__ (self, n_points, d_model, in_hidden_size = 256, dropout = 0.2):
        super().__init__()

        # nn.Linear ?
        self.q = MLP(n_points, d_model, in_hidden_size)
        self.k = MLP(n_points, d_model, in_hidden_size)
        self.v = MLP(n_points, d_model, in_hidden_size)
        self.mh = nn.MultiheadAttention(d_model, 1, dropout=dropout, batch_first=True)


    def forward (self, x):
        _, w = self.mh(self.q(x), self.k(x), self.v(x))
        return w

    
class Model (nn.Module):
    
    def __init__(self, n_points, n_species, hidden_size, d_model, dropout=0):
        self.encoder = TemporalEncoder(n_points, n_species, hidden_size, dropout=dropout)
        self.decoder = TemporalEncoder(n_species, n_points, hidde_size, dropout=dropout)
        self.adj = GraphGenerator(n_points, d_model, hidden_size, dropout=dropout)

    def forward (self, x):
        adj = self.adj(x)
        x = self.encoder(x)
        x = F.relu(adj*x)
        x = self.decoder(x)
        return x


