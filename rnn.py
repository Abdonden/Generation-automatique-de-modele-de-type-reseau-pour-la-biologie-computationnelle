#rnn originale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda
#device = torch.device("cpu")

writer = SummaryWriter(comment="")

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


# === Modèle GCN avec paramètre W ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels=512, out_channels=512, num_embeddings=5, hidden_embedder_size = 512, hidden_size = 512):#num_embeddings : nombre de vecteurs par espèce (5).
        super().__init__()
        self.embedder=Embedder(in_channels, out_channels, hidden_embedder_size)
        self.conv1 = GCNConv(out_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        # Paramètre entraînable W pour la fonction de score
        self.W = nn.Parameter(torch.randn(out_channels))

        self.lin_in = nn.Linear(out_channels, hidden_size)
        self.lin_out = nn.Linear(hidden_size, 1)

    def forward(self, x, edge_index, src_idx, dst_idx):


        #aggregation des trajectoires
        x = F.relu(self.embedder(x)) 
        y = x
        input = x
        y = F.relu(self.conv1(y, edge_index))
        y = self.conv2(y, edge_index)
        x = F.relu(x+y)
        y =x



        embeddings = x+input


        src_emb, dst_emb = embeddings[src_idx], embeddings[dst_idx]
        pred = dst_emb - src_emb
        pred = self.lin_in(pred)
        pred = F.relu(pred)
        pred = self.lin_out(pred)
        return pred


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
src = 10
dst = src+1
dataset = torch.load("dataset_sat.pt")#[:20]
batch_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size)

#examples statistics
nb_positive = torch.tensor(3500)
nb_negative = 1854 
pos_weight = nb_negative/nb_positive

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# === Instanciation du modèle et optimiseur ===
model_gcn = GCN(out_channels = 512)
model_gcn.to(device)
#sat_mdl = SatMLP(num_trajectoires=5).to(device)

model = model_gcn

#optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-8)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)






# === Perte avant entraînement ===
#loss_before = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
#print(f"\n📊 Perte moyenne avant entraînement : {loss_before:.4f}")


n=0

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

#        embeddings = model_gcn(feats,edges)
        #scores = [f(embeddings[i1], embeddings[i2], model_gcn.W) for i1, i2 in interaction_indices]
        #scores_tensor = torch.stack(scores)
        #scores_tensor = f (embeddings[interaction_indices[:,0]], embeddings[interaction_indices[:,1]], model_gcn.W)

        src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination

        prediction = model (feats,edges, src_idx, dst_idx)



        loss = criterion(prediction.squeeze() , labels.float())


        #labels = torch.ones_like(scores_tensor)
        #loss = criterion(scores_tensor, labels)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        total_loss += loss
        writer.add_scalar("batch/train", loss, n)
        n+=1

    avg_loss = total_loss / len(dataloader)
    #avg_loss.backward()
    #optimizer.step()
    #scheduler.step(avg_loss)

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
        print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f}")







# === Perte après entraînement ===
loss_after = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne après entraînement : {loss_after:.4f}")
