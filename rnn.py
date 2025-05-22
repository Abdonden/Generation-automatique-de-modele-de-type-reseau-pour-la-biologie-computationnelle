#rnn originale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from generate_dataset import MyData


device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda
#device = torch.device("cpu")

writer = SummaryWriter(comment="")

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
    def __init__ (self, n_points, output_size, hidden_size):
        super().__init__()

        self.lin_in = nn.Linear(n_points, hidden_size)
        self.hid = nn.Linear(hidden_size, hidden_size)
        self.lin_out = nn.Linear(hidden_size, output_size)

        self.has_proj = n_points > hidden_size
        if self.has_proj:
            self.proj = nn.Linear(n_points, hidden_size)

        
    
    def forward(self, x):


        skip = x

        x = self.lin_in(x)
        x = F.gelu(x)
        x = self.hid(x)

        if self.has_proj:
            skip = self.proj(skip)
        
        x = skip+F.gelu(x)
        x = self.lin_out(x)
        return x

class Embedder (torch.nn.Module):
    def __init__(self, n_points, output_size, hidden_size):
        super().__init__()
        self.mlp = MLP(n_points, n_points, hidden_size)
        self.rho = MLP(n_points, output_size, hidden_size)
    def forward(self,x):
        x = F.gelu(self.mlp(x))
        return self.rho(x.sum(-2))

class GCNResnet(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features < out_features:
            raise ValueError (f"GCNResnet can only handle layers with in_features >= out_features. Got in_features={in_features} and out_features={out_features}")
        self.has_proj  = in_features != out_features
        self.conv = SAGEConv(in_features, out_features)
        if self.has_proj:
            self.proj = nn.Linear(in_features, out_features)

    
    def forward(self, x, edge_index):
        y = self.conv (x,edge_index)
        if self.has_proj:
            skip = self.proj(x)
        else:
            skip = x
        return skip+F.gelu(y)

# === Modèle GCN avec paramètre W ===
class GCN(torch.nn.Module):
    def __init__(self, nb_points=512, out_channels=64):#num_embeddings : nombre de vecteurs par espèce (5).
        self.n_pts = nb_points

        super().__init__()
        #self.transformer = PermutationInvariantTransformer(nb_points, out_channels, out_channels, 1, 1, dropout=0) 
        self.embedder=Embedder(nb_points, out_channels, out_channels)
        self.conv1 = GCNResnet(out_channels, out_channels)
        self.conv2 = GCNResnet(out_channels, out_channels)
        self.conv3 = GCNResnet(out_channels, out_channels)

        self.bn = nn.BatchNorm1d(out_channels)
        self.lin = nn.Sequential (MLP(out_channels*3, out_channels*2, out_channels*2),
                                  nn.ReLU(),
                                  MLP(out_channels*2, 1, out_channels*2))

    def forward(self,batch, x, edge_index, src_idx, dst_idx):

        #ts = self.transformer(x)

        #x = ts
        x = self.embedder(x)



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

        pred = self.lin(xy) - self.lin(yx)
        return pred



def create_query (src_emb, dst_emb, graph_emb):
    return torch.cat((src_emb, dst_emb, graph_emb), dim=-1)


def test_model(model, dloader):
    n_examples = 0
    tot_loss = 0
    with torch.no_grad():
        for batch in dloader:
            feats, edges = batch.x, batch.edge_index
            feats, edges = feats.to(device).float(), edges.to(device)

            interaction_indices= batch.y.to(device)
            n_batched_examples = interaction_indices.shape[0]
            
            
            labels = batch.labels.to(device)
            labels = labels.long()

            src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination

            b = batch.batch.to(device)

            prediction = model (b,feats,edges, src_idx, dst_idx)



            loss = criterion(prediction.squeeze() , labels.float())
            tot_loss += loss.detach()*n_batched_examples
            n_examples += n_batched_examples
    return tot_loss/n_examples
            



    


# === Chargement du dataset ===
src = 0
dst = src+1
dataset = torch.load("dataset_sat.pt")
n_datas = len(dataset)
trainset = dataset[:int(n_datas*0.9)]
testset = dataset[int(n_datas*0.9):]
batch_size = len(trainset)
dataloader = DataLoader(trainset, batch_size=batch_size) #, shuffle=True)
testloader = DataLoader(testset, batch_size=len(testset))

#examples statistics
nb_positive = torch.tensor(1854)
nb_negative = 1854 
pos_weight = nb_negative/nb_positive

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# === Instanciation du modèle et optimiseur ===
model_gcn = GCN(out_channels = 16)
model_gcn.to(device)

model = model_gcn

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
#scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=100) 
n=0
best=10000
best_test = torch.tensor(10000)

for i,epoch in enumerate(range(50000000000)):
    model.train()

    total_loss = 0.0
    n_examples = 0
    for j,batch in enumerate(dataloader):
        optimizer.zero_grad()
        feats, edges = batch.x, batch.edge_index
        feats, edges = feats.to(device).float(), edges.to(device)

        interaction_indices= batch.y.to(device)
        n_batched_examples = interaction_indices.shape[0]
        
        
        labels = batch.labels.to(device)
        labels = labels.long()

        src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination

        b = batch.batch.to(device)

        prediction = model (b,feats,edges, src_idx, dst_idx)



        loss = criterion(prediction.squeeze() , labels.float())


        loss.backward()
        old_lr = optimizer.param_groups[0]['lr']
        #scheduler.step(loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print (f"[learning rate]: {old_lr} -> {new_lr}")
        optimizer.step()
        loss = loss.item()
        total_loss += loss*n_batched_examples
        n_examples += n_batched_examples
        writer.add_scalar("batch/train", loss, n)
        n+=1

    avg_loss = total_loss / n_examples
    if avg_loss < best:
        best=avg_loss
    writer.add_scalar("epoch/train", avg_loss, i)

    test_loss = test_model(model, testloader)
    writer.add_scalar("epoch/test", test_loss, i)
    if best_test > test_loss:
        best_test = test_loss


    # Logs
    if epoch % 500 == 0:
        print(f"\n🔧 Norme des gradients à l'époque {epoch} :")
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f" - {name} grad norm: {grad_norm:.6f}")
        print(f" - Total gradient norm: {total_grad_norm:.6f}")

    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f} best={best} best_test={best_test}")
    print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f}")







# === Perte après entraînement ===
loss_after = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne après entraînement : {loss_after:.4f}")
