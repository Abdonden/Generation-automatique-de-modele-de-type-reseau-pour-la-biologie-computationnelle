
import torch
import torch.nn as nn
import torch.nn.functional as F




class Embedder (nn.Module):
    def __init__(self, embedding_size = 256, nhead = 4, num_layers=1):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead,
                                                  dropout=0)

    def forward(self, x):
        x = self.linear(x)
        x = self.encoder(x)
        #return x.max(dim=-2)[0]
        return F.relu(x.mean(dim=1))
