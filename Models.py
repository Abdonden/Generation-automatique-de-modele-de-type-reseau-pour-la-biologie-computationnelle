import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool, global_max_pool
from generate_dataset import MyData
from edge_prediction import *

from transformers import LongformerConfig, LongformerModel
from embedding import *




class GCNResnet(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.2):
        super().__init__()
        if in_features < out_features:
            raise ValueError (f"GCNResnet can only handle layers with in_features >= out_features. Got in_features={in_features} and out_features={out_features}")
        self.has_proj  = in_features != out_features
        self.conv = SAGEConv(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        if self.has_proj:
            self.proj = nn.Linear(in_features, out_features)

    
    def forward(self, x, edge_index):
        y = self.conv (x,edge_index)
        y = self.dropout(y)
        if self.has_proj:
            skip = self.proj(x)
        else:
            skip = x
        return skip+F.gelu(y)

# === Modèle GCN avec paramètre W ===
class GCN(torch.nn.Module):
    def __init__(self, nb_points=512, out_channels=64, dropout=0.2):#num_embeddings : nombre de vecteurs par espèce (5).
        self.n_pts = nb_points

        super().__init__()
        #self.embedder= PermutationInvariantTransformer(1, out_channels, out_channels, 4, 2, dropout=dropout) 
        self.embedder = TransformerEmbedder(nb_points, out_channels, out_channels, dropout=dropout, num_layers=1, nhead=1)
        #self.embedder=Embedder(nb_points, out_channels, out_channels*2, dropout=dropout)
        self.conv1 = GCNResnet(out_channels, out_channels, dropout=dropout)
        self.conv2 = GCNResnet(out_channels, out_channels, dropout=dropout)
        self.conv3 = GCNResnet(out_channels, out_channels, dropout=dropout)

        self.bn = nn.BatchNorm1d(out_channels)
        self.lin = nn.Sequential (MLP(out_channels*3, out_channels*2, out_channels*2, dropout=dropout),
                                  nn.ReLU(),
                                  MLP(out_channels*2, 1, out_channels*2, dropout=dropout))


    def forward(self,batch, src_idx, dst_idx):

        #ts = self.transformer(x)

        #x = ts

        x, edge_index, mask, times = batch.x, batch.edge_index, batch.masks, batch.times

        if x.isnan().any():
            raise ValueError ("x is nan before embeddings")



        x = self.embedder(x,mask, times)


        if x.isnan().any():
            raise ValueError ("x is nan after embeddings")

        #passage au GCN
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x,edge_index)

        if x.isnan().any():
            raise ValueError ("x is nan after convolutions")

        #x = self.bn(x)

        node_embeddings = x

        graph_emb = global_max_pool(x, batch.batch) 
        #graph_emb = global_mean_pool(x, batch) 
        graph_emb = graph_emb[batch.batch] #graph embedding pour chaque sommet du graphe

        src_emb, dst_emb = node_embeddings[src_idx], node_embeddings[dst_idx]
        graph_emb = graph_emb[src_idx] #get the graph_emb of the source node (is the same as the dst node or every other node in the graph)
        
        xy = create_query(src_emb, dst_emb, graph_emb)
        yx = create_query(dst_emb, src_emb, graph_emb)

        predxy = self.lin(xy)
        predyx = self.lin(yx)



        return node_embeddings, (predxy, predyx), xy



def create_query (src_emb, dst_emb, graph_emb):
    return torch.cat((src_emb, dst_emb, graph_emb), dim=-1)


