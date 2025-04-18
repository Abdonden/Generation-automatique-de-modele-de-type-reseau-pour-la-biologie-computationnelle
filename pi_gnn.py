import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

class PiGNNEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_max, hidden_nodes, n_layers, use_dustbins=True, features='gcn', dropout=0.5, n_classes=256):
        super(PiGNNEmbedding, self).__init__()
        self.n_max = n_max
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.hidden_nodes = hidden_nodes
        self.use_dustbins = use_dustbins
        self.features = features
        self.dropout = dropout
        self.n_classes = n_classes

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.conv_layers.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))

        self.fc = nn.Linear(hidden_dim, n_classes)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout_layer(x)
        x = self.fc(x)
        return x  # Shape: (num_nodes, n_classes)

