
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import *

class MLP (torch.nn.Module):
    def __init__ (self, n_points, output_size, hidden_size, dropout=0.2):
        super().__init__()

        self.lin_in = nn.Linear(n_points, hidden_size)
        self.hid = nn.Linear(hidden_size, hidden_size)
        self.lin_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.has_proj = n_points > hidden_size
        if self.has_proj:
            self.proj = nn.Linear(n_points, hidden_size)

        
    
    def forward(self, x):


        skip = x

        x = self.lin_in(x)
        x = F.gelu(x)
        x = self.hid(x)
        x = self.dropout(x)

        if self.has_proj:
            skip = self.proj(skip)
        
        x = skip+F.gelu(x)
        x = self.lin_out(x)
        return x



class TransformerEmbedder (nn.Module):
    def __init__(self, n_pts, d_model, output_size, nhead=1, dropout=0, num_layers=4):
        super().__init__()
        self.lin_in = nn.Linear(1, d_model)
        self.time_in = nn.Linear(1,d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lin_out = nn.Linear(d_model, output_size)

    def forward(self, x, mask, times):
        n_nodes, n_traj, n_pts = x.size()
        #times = times.unsqueeze(0).unsqueeze(-1)

        inputs = x
        times = times.unsqueeze(-1)
        times = self.time_in(times)

        if times.isnan().any():
            raise ValueError("time is nan after projection")

        x = x.unsqueeze(-1)
        x = self.lin_in(x)
        if x.isnan().any():
            raise ValueError("x is nan after projection")


        x = x+times

        x = x.reshape(n_nodes*n_traj, n_pts,-1)
        mask = mask.reshape(n_nodes*n_traj, n_pts)

        feats = x
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        if x.isnan().any():
            print (mask.shape, x.shape)
            print (mask.all(dim=-1).any())
            print (feats.min(), feats.max(), inputs.min(), inputs.max())
            print (times.min(), times.max())
            raise ValueError(f"x is nan after transformer: masked={mask.any(), mask.all()}")


        x = x.mean(dim=1)

        x = x.reshape(n_nodes,n_traj,-1)
        x = x.mean(dim=1)
        x = self.lin_out(x)
        return x



#mdl = TransformerEmbedder(512,64,10)
#data = torch.load("dataset_sat.pt")[0]
