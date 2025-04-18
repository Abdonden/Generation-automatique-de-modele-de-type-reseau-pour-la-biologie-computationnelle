import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Définir l'embedder
class Embedder(nn.Module):
    def __init__(self, embedding_size=256, nhead=4, num_layers=1):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)
        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead, dropout=0)

    def forward(self, x):
        # Passer par la couche linéaire
        x = self.linear(x)
        # Passer par le Transformer Encoder
        x = self.encoder(x)
        # Moyenne et Relu pour réduction de dimension
        return F.relu(x.mean(dim=1))  # [num_nodes, embedding_size]

# Définir le modèle GCN
class GCN(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        self.embedder = Embedder(embedding_size)
        self.conv1 = GCNConv(embedding_size, 512)
        self.conv2 = GCNConv(512, 512)
        self.conv3 = GCNConv(512, 512)
        self.conv4 = GCNConv(512, 512)

    def forward(self, data):
        # Extraire les features et l'edge_index de chaque graphe
        x, edge_index = data["features"], data["edges"]
        
        # Ajouter une dimension pour correspondre à l'input attendu par l'embedder
        x = x.unsqueeze(-1)  # [num_nodes, 1]
        
        # Passer les features à travers l'embedder pour obtenir des embeddings fixes
        x = self.embedder(x)
        
        # Passer les embeddings à travers les couches GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        
        return x  # [num_nodes, hidden_size] : Embedding final des nœuds

# Charger le dataset depuis 'dataset_complet.pt'
batch = torch.load("dataset_complet.pt")

# Initialiser le modèle GCN
model = GCN(embedding_size=256)

# Liste pour stocker les features finales de tous les graphes
final_species_features = []

# Appliquer GCN à chaque graphe dans le batch
for i in range(len(batch["model_id"])):
    # Extraire les features et l'edge_index pour le modèle courant
    features = batch["features"][i]  # [num_nodes, feature_dim]
    edge_index = batch["edges"][i]   # [2, num_edges]
    
    # Convertir les features en float32 si nécessaire
    features = features.float()
    
    # Créer un dictionnaire de données pour le graphe
    data = {
        "features": features,
        "edges": edge_index
    }
    
    # Passer les données dans le modèle GCN
    output = model(data)
    
    # Ajouter les résultats (features des nœuds après GCN) à la liste des résultats
    final_species_features.append(output)

    # Enregistrer les features finales pour chaque graphe dans un fichier
    feature_file_name = f"feature_finaux_{batch['model_id'][i]}.pt"
    torch.save(output, feature_file_name)
    print(f"Features du modèle {batch['model_id'][i]} enregistrées dans {feature_file_name}")

# À ce stade, final_species_features contient les features des espèces pour tous les graphes
# Les features pour chaque espèce sont des embeddings appris pour chaque nœud dans chaque graphe
print(f"\n📊 Features des espèces pour tous les graphes :")
for i, species_feature in enumerate(final_species_features):
    print(f"\n🔢 Modèle {batch['model_id'][i]}")
    print(f" - Shape des features après GCN: {species_feature.shape}")
