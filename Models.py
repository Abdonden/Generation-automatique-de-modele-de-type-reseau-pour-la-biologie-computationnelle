import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool, global_max_pool
from generate_dataset import MyData
from edge_prediction import *




class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len=512, input_dim)
        _, h_n = self.gru(x)  # h_n shape: (num_layers, batch_size, hidden_dim)
        h_last = h_n[-1]      # shape: (batch_size, hidden_dim)
        out = self.proj(h_last)  # shape: (batch_size, output_dim)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0,1)
        return self.dropout(x)

class PermutationInvariantTransformer(nn.Module):
    def __init__(self, num_features, d_model, output_size, nhead, num_layers, dropout=0.1):
        """
        Transformer qui est invariant aux permutations des séries temporelles.
        
        :param input_dim: La dimension de chaque élément dans la séquence (par exemple, nombres de pas de temps)
        :param d_model: La dimension du modèle interne
        :param nhead: Le nombre de têtes d'attention
        :param num_layers: Le nombre de couches de l'encodeur Transformer
        :param num_series: Le nombre de séries temporelles indépendantes
        :param dropout: Taux de dropout pour la régularisation
        """
        super(PermutationInvariantTransformer, self).__init__()


        # Embedding pour chaque série (input_dim -> d_model)
        self.embedding = nn.Linear(num_features, d_model)

        # Positionnal Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Couche de sortie pour obtenir une représentation finale
        self.fc_out = nn.Linear(d_model, output_size) 

    def forward(self, x):
        # Embedding des séries temporelles
        x = self.embedding(x)  # (batch_size * num_series, seq_len, d_model)

        # Ajouter les encodages de position
        x = self.positional_encoding(x)  # (batch_size * num_series, seq_len, d_model)


        # Passer dans l'encodeur Transformer
        x = self.transformer_encoder(x)  # (seq_len, batch_size * num_series, d_model)


        # Retourner à la forme originale : (batch_size * num_series, d_model)
        x = x.mean(dim=1)  # Calculer la moyenne sur la séquence (mean pooling)


        # Passer par la couche linéaire de sortie
        x = self.fc_out(x)  # (batch_size * num_series, 1)


        return x


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

class Embedder (torch.nn.Module):
    def __init__(self, n_points, output_size, hidden_size, dropout=0.2):
        super().__init__()
        self.mlp = MLP(n_points, n_points, hidden_size)
        self.dropout  = nn.Dropout(dropout)
        self.rho = MLP(n_points, output_size, hidden_size)
    def forward(self,x):
        x = F.gelu(self.mlp(x))
        x = self.dropout(x)
        return self.rho(x.sum(-2))

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
        #self.embedder= PermutationInvariantTransformer(nb_points, out_channels, out_channels, 4, 2, dropout=dropout) 
        self.embedder=Embedder(nb_points, out_channels, out_channels*2, dropout=dropout)
        self.conv1 = GCNResnet(out_channels, out_channels, dropout=dropout)
        self.conv2 = GCNResnet(out_channels, out_channels, dropout=dropout)
        self.conv3 = GCNResnet(out_channels, out_channels, dropout=dropout)

        self.bn = nn.BatchNorm1d(out_channels)
        self.lin = nn.Sequential (MLP(out_channels*3, out_channels*2, out_channels*2, dropout=dropout),
                                  nn.ReLU(),
                                  MLP(out_channels*2, 1, out_channels*2, dropout=dropout))


    def forward(self,batch, x, edge_index, src_idx, dst_idx):

        #ts = self.transformer(x)

        #x = ts
        x = self.embedder(x)
        ts = x.clone()



        #passage au GCN
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x,edge_index)

        #x = self.bn(x)

        node_embeddings = x

        graph_emb = global_max_pool(x, batch) 
        #graph_emb = global_mean_pool(x, batch) 
        graph_emb = graph_emb[batch] #graph embedding pour chaque sommet du graphe

        src_emb, dst_emb = node_embeddings[src_idx], node_embeddings[dst_idx]
        graph_emb = graph_emb[src_idx] #get the graph_emb of the source node (is the same as the dst node or every other node in the graph)
        
        xy = create_query(src_emb, dst_emb, graph_emb)
        yx = create_query(dst_emb, src_emb, graph_emb)

        predxy = self.lin(xy)
        predyx = self.lin(yx)



        return node_embeddings, (predxy, predyx), xy



def create_query (src_emb, dst_emb, graph_emb):
    return torch.cat((src_emb, dst_emb, graph_emb), dim=-1)


