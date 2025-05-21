#rnn originale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda
#device = torch.device("cpu")

writer = SummaryWriter(comment="")

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


class Embedder (torch.nn.Module):
    def __init__ (self, n_points, output_size, hidden_size):
        super().__init__()

        self.lin_in = nn.Linear(n_points, hidden_size)
        self.lin_out = nn.Linear(hidden_size, output_size)
        self.rho = nn.Linear(output_size,  output_size)
    
    def forward(self, x):




        x = self.lin_in(x)
        x = F.relu(x)
        x = self.lin_out(x)
        x = x.sum(-2)
        return self.rho(x)

class GCNResnet(torch.nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.conv_in = GCNConv(n_features, hidden_size)
        self.conv_out = GCNConv(hidden_size, n_features)
    
    def forward(self, x, edge_index):
        y = x
        y = self.conv_in(y,edge_index)
        y = F.leaky_relu(y)
        y = self.conv_out(y,edge_index)
        
        return x+y

# === Modèle GCN avec paramètre W ===
class GCN(torch.nn.Module):
    def __init__(self, nb_points=512, out_channels=512, num_embeddings=5, hidden_embedder_size = 512, hidden_size = 512):#num_embeddings : nombre de vecteurs par espèce (5).
        self.n_pts = nb_points
        self.n_trajectories = num_embeddings

        super().__init__()
        self.transformer = PermutationInvariantTransformer(nb_points, out_channels,out_channels, 8, 3, dropout=0) 
        #self.embedder=Embedder(nb_points, out_channels, hidden_embedder_size)
        self.conv_in = GCNConv (nb_points*num_embeddings, out_channels)
        self.conv1 = GCNResnet(out_channels, out_channels)
        self.conv2 = GCNResnet(out_channels, out_channels)
        self.conv3 = GCNResnet(out_channels, out_channels)

        self.lin_in = nn.Linear(out_channels*2, hidden_size)
        self.lin_hid = nn.Linear(hidden_size, hidden_size)
        #self.lin_out = nn.Linear(hidden_size, 1)
        self.lin_out = nn.Linear(out_channels*2, 1)

    def forward(self, x, edge_index, src_idx, dst_idx):

        ts = self.transformer(x)

        #aggregation des trajectoires
        #x = F.relu(self.embedder(x)) 
        x = x.reshape(-1, self.n_pts*self.n_trajectories)
        x = self.conv_in(x, edge_index)
        x = F.leaky_relu(x)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))



        embeddings = torch.cat ((x, ts), dim=-1)


        src_emb, dst_emb = embeddings[src_idx], embeddings[dst_idx]
        pred = dst_emb - src_emb
        #pred = pred.sum(-1)
        #pred = F.relu(self.lin_in(pred))
        #pred = F.relu(self.lin_hid(pred))
        pred = self.lin_out(pred)
        return pred

# === Chargement du dataset ===
src = 0
dst = src+1
dataset = torch.load("dataset_sat.pt")
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

#examples statistics
nb_positive = torch.tensor(1854)
nb_negative = 1854 
pos_weight = nb_negative/nb_positive

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# === Instanciation du modèle et optimiseur ===
model_gcn = GCN(out_channels = 512, num_embeddings=10, hidden_size=2048)
model_gcn.to(device)

model = model_gcn

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=100) 
n=0
best=10000

for i,epoch in enumerate(range(50000000000)):
    model.train()

    total_loss = 0.0
    for j,batch in enumerate(dataloader):
        optimizer.zero_grad()
        feats, edges = batch.x, batch.edge_index
        feats, edges = feats.to(device).float(), edges.to(device)

        target = batch.y.to(device)
        
        interaction_indices = target[:,:2]
        labels = target[:,2]
        labels = labels.long()

        src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination

        prediction = model (feats,edges, src_idx, dst_idx)



        loss = criterion(prediction.squeeze() , labels.float())


        loss.backward()
        old_lr = optimizer.param_groups[0]['lr']
        #scheduler.step(loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print (f"[learning rate]: {old_lr} -> {new_lr}")
        optimizer.step()
        loss = loss.item()
        total_loss += loss
        writer.add_scalar("batch/train", loss, n)
        if loss < best:
            best=loss
        n+=1

    avg_loss = total_loss / len(dataloader)

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
        print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f} best={best}")
    #print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f}")







# === Perte après entraînement ===
loss_after = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne après entraînement : {loss_after:.4f}")
