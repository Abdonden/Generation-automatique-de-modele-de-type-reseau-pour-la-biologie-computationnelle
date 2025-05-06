#rnn originale
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# === Modèle GCN avec paramètre W ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels=256, out_channels=512, num_embeddings=5):#num_embeddings : nombre de vecteurs par espèce (5).
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, out_channels)
        self.conv3 = GCNConv(out_channels, out_channels)
        self.conv4 = GCNConv(out_channels, out_channels)

        # Paramètre entraînable W pour la fonction de score
        self.W = nn.Parameter(torch.randn(num_embeddings, out_channels))

    def forward(self, x, edge_index):
        # x: (N, 5, 256) -> flatten to (N*5, 256)    Aplatis x de (N, 5, 256) à (N*5, 256) pour appliquer la GCN.
        x = x.view(-1, x.size(-1))  # (N*5, 256)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)
        # Output: Reforme x de (N*5, 512) vers (N, 5, 512) → pour obtenir 5 embeddings par espèce.
        x_out = x.view(-1, 5, x.size(-1))  
        return x_out

# === Fonction de score f ===
def f(x_i, x_j, W):
    diff = x_i - x_j            # (5, 512)
    weighted_diff = W * diff    # (5, 512)
    summed = torch.sum(weighted_diff)  # scalaire
    return torch.sigmoid(summed)       # entre 0 et 1

# === Fonction d’évaluation ===
def evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion):
    model_gcn.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, interaction_indices in zip(batch_data, interaction_indices_batch):
            embeddings = model_gcn(data["features"], data["edges"])  # (N, 5, 512)
            scores = [f(embeddings[i1], embeddings[i2], model_gcn.W) for i1, i2 in interaction_indices]
            scores_tensor = torch.stack(scores)
            labels = torch.ones_like(scores_tensor)
            loss = criterion(scores_tensor, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(batch_data)
    return avg_loss

# === Chargement du dataset ===
batch = torch.load("dataset_complet.pt", weights_only=True)
criterion = nn.BCELoss()
num_models = len(batch["model_id"])

# === Instanciation du modèle et optimiseur ===
model_gcn = GCN()
optimizer = torch.optim.Adam(model_gcn.parameters(), lr=0.000001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, verbose=True)
# === Préparation des données ===
batch_data = []
interaction_indices_batch = []

for graph_idx in range(num_models):
    features = batch["features"][graph_idx] 
    edge_index = batch["edges"][graph_idx]

    features = torch.stack(features, dim=0) 
    num_nodes = features.size(0)

    if edge_index.max() >= num_nodes * 5:
        print(f"⚠️ Graphe {batch['model_id'][graph_idx]} : index max {edge_index.max()} >= nb nœuds {num_nodes*5}, filtrage")
        mask = (edge_index[0] < num_nodes * 5) & (edge_index[1] < num_nodes * 5)
        edge_index = edge_index[:, mask]

    data = {"features": features, "edges": edge_index}
    interaction_path = f"interactions/interaction_indices_{batch['model_id'][graph_idx]}.pt"
    raw_indices = torch.load(interaction_path, weights_only=True)
    interaction_indices = [(int(i1), int(i2)) for i1, i2 in raw_indices if int(i1) < num_nodes and int(i2) < num_nodes]

    batch_data.append(data)
    interaction_indices_batch.append(interaction_indices)

# === Perte avant entraînement ===
loss_before = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne avant entraînement : {loss_before:.4f}")

# === Entraînement avec Early Stopping ===
early_stop_threshold = 0.001
patience = 5000
best_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(200000):
    model_gcn.train()
    optimizer.zero_grad()

    total_loss = 0.0
    for data, interaction_indices in zip(batch_data, interaction_indices_batch):
        embeddings = model_gcn(data["features"], data["edges"])
        scores = [f(embeddings[i1], embeddings[i2], model_gcn.W) for i1, i2 in interaction_indices]
        scores_tensor = torch.stack(scores)
        labels = torch.ones_like(scores_tensor)
        loss = criterion(scores_tensor, labels)
        total_loss += loss

    avg_loss = total_loss / num_models
    avg_loss.backward()
    optimizer.step()
    scheduler.step(avg_loss)

    # Logs
    if epoch % 500 == 0:
        print(f"\n🔧 Norme des gradients à l'époque {epoch} :")
        total_grad_norm = 0.0
        for name, param in model_gcn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f" - {name} grad norm: {grad_norm:.6f}")
        print(f" - Total gradient norm: {total_grad_norm:.6f}")

    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Perte moyenne : {avg_loss.item():.4f}")

    if avg_loss.item() < early_stop_threshold:
        print(f"\n✅ Seuil de perte atteint (Loss < {early_stop_threshold}), arrêt.")
        break

    if avg_loss.item() < best_loss:
        best_loss = avg_loss.item()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"\n✅ Pas d'amélioration depuis {patience} époques, arrêt.")
        break

# === Perte après entraînement ===
loss_after = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne après entraînement : {loss_after:.4f}")
