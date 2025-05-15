#rnn originale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.loader import DataLoader


device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda
#device = torch.device("cpu")

class Embedder (torch.nn.Module):
    def __init__ (self, n_points, n_embeddings, hidden_size):
        super().__init__()
        self.lin_in = nn.Linear(n_points, hidden_size)
        self.lin_out = nn.Linear(hidden_size, n_embeddings)
        self.rho = nn.Linear(n_embeddings,  n_embeddings)
    
    def forward(self, x):
        x = self.lin_in(x)
        x = F.relu(x)
        x = self.lin_out(x)
        x = x.sum(-2)
        return self.rho(x)


# === Modèle GCN avec paramètre W ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels=512, out_channels=512, num_embeddings=5, hidden_embedder_size = 512):#num_embeddings : nombre de vecteurs par espèce (5).
        super().__init__()
        self.embedder=Embedder(in_channels, out_channels, hidden_embedder_size)
        self.conv1 = GCNConv(out_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        # Paramètre entraînable W pour la fonction de score
        self.W = nn.Parameter(torch.randn(out_channels))

    def forward(self, x, edge_index):


        #aggregation des trajectoires
        x = F.relu(self.embedder(x)) 
        y = x
        input = x
        y = F.relu(self.conv1(y, edge_index))
        y = self.conv2(y, edge_index)
        x = F.relu(x+y)
        y =x
        return x+input

class SatMLP (torch.nn.Module):
    def __init__(self,  num_trajectoires=5, in_channels=512, out_channels=512, hidden_size=1024):
        super().__init__()
        self.num_trajectoires = num_trajectoires
        self.in_channels = in_channels
        self.lin_in = nn.Linear(num_trajectoires*in_channels, hidden_size)
        self.lin_out = nn.Linear(hidden_size, out_channels)
    
    def forward (self, x):
        x = x.reshape(-1, self.num_trajectoires*self.in_channels)
        x = self.lin_in(x)
        x = F.relu(x)
        x = self.lin_out(x)
        return F.relu(x)

# === Fonction de score f ===
def f(x_i, x_j, W):
    diff = x_i - x_j            # (512,)
    #weighted_diff = W.dot(diff)
    return F.sigmoid(diff.sum())    
    #return torch.sigmoid(weighted_diff)
# === Fonction d’évaluation ===
def evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion):
    model_gcn.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, interaction_indices in zip(batch_data, interaction_indices_batch):
            feats, edges = data["features"], data["edges"]
            feats, edges = feats.to(device), edges.to(device)
            interaction_indices = torch.tensor(interaction_indices).to(device)
            embeddings = model_gcn(feats, edges)  # (N, 5, 512)
            scores = [f(embeddings[i1], embeddings[i2], model_gcn.W) for i1, i2 in interaction_indices]
            scores_tensor = torch.stack(scores)
            labels = torch.ones_like(scores_tensor)
            loss = criterion(scores_tensor, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(batch_data)
    return avg_loss

# === Chargement du dataset ===
dataset = torch.load("dataset_sat.pt")[3:4]
dataloader = DataLoader(dataset, batch_size=len(dataset))

criterion = nn.BCEWithLogitsLoss(reduction="mean")


# === Instanciation du modèle et optimiseur ===
#model_gcn = GCN()
#model_gcn.to(device)
sat_mdl = SatMLP(num_trajectoires=5).to(device)

#optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-6)
optimizer = torch.optim.Adam(sat_mdl.parameters(), lr=1e-6)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)




# === Perte avant entraînement ===
#loss_before = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
#print(f"\n📊 Perte moyenne avant entraînement : {loss_before:.4f}")



for epoch in range(50000000000):
    #model_gcn.train()

    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        feats, edges = batch.x, batch.edge_index
        feats, edges = feats.to(device).float(), edges.to(device)

        interaction_indices = batch.y.to(device)

#        embeddings = model_gcn(feats,edges)
        embeddings = sat_mdl(feats)
        #scores = [f(embeddings[i1], embeddings[i2], model_gcn.W) for i1, i2 in interaction_indices]
        #scores_tensor = torch.stack(scores)
        #scores_tensor = f (embeddings[interaction_indices[:,0]], embeddings[interaction_indices[:,1]], model_gcn.W)

        src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination
        src_emb, dst_emb = embeddings[src_idx], embeddings[dst_idx]

        tgt1 = (dst_emb - src_emb).sum(1)
        tgt2 = (src_emb - dst_emb).sum(1)


        labels1 = torch.ones(tgt1.shape).to(device)
        labels2 = torch.zeros(tgt2.shape).to(device)

        tgt = torch.cat((tgt1,tgt2))
        labels = torch.cat((labels1,labels2))

        loss = criterion(tgt, labels)

        #labels = torch.ones_like(scores_tensor)
        #loss = criterion(scores_tensor, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    #avg_loss.backward()
    #optimizer.step()
    #scheduler.step(avg_loss)

    # Logs
    if epoch % 500 == 0:
        print(f"\n🔧 Norme des gradients à l'époque {epoch} :")
        total_grad_norm = 0.0
        for name, param in sat_mdl.named_parameters():
#        for name, param in model_gcn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f" - {name} grad norm: {grad_norm:.6f}")
        print(f" - Total gradient norm: {total_grad_norm:.6f}")

    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f}")







# === Perte après entraînement ===
loss_after = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne après entraînement : {loss_after:.4f}")
