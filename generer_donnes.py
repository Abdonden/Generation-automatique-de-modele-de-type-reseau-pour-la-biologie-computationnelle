import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Définition du modèle Embedder
class Embedder(nn.Module):
    def __init__(self, embedding_size=256, nhead=2):
        super().__init__()
        self.linear = nn.Linear(1, embedding_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=nhead, dropout=0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        outputs = []
        for traj in x:
            traj = traj.float()
            embedded = self.linear(traj)            # (seq_len, embedding_size)
            embedded = embedded.unsqueeze(1)        # (seq_len, 1, embedding_size)
            encoded = self.encoder(embedded)        # (seq_len, 1, embedding_size)
            pooled = F.relu(encoded.mean(dim=0))    # (1, embedding_size)
            outputs.append(pooled.squeeze(0))       # (embedding_size,)
        return outputs

# Répertoires
input_dir = 'features'
output_dir = 'featuress'
os.makedirs(output_dir, exist_ok=True)
# Suppression des fichiers existants dans le répertoire 'features' pour éviter les doublons
for fichier in os.listdir(output_dir):
    if fichier.endswith(".pt"):
        os.remove(os.path.join(output_dir, fichier))
# Chargement du modèle
model = Embedder()

# Traitement de tous les fichiers
fichiers = [f for f in os.listdir(input_dir) if f.endswith('.pt')]
for nom_fichier in fichiers:
    print(f"Traitement de : {nom_fichier}")
    chemin = os.path.join(input_dir, nom_fichier)
    
    # Charger les données
    features = torch.load(chemin, weights_only=True)
    n_trajectoires = len(features)
    n_especes = features[0].shape[0]

    # Réorganiser les trajectoires par espèce
    especes = [[] for _ in range(n_especes)]
    for traj in features:
        for i in range(n_especes):
            espece_i = traj[i].unsqueeze(1)
            especes[i].append(espece_i)

    # Appliquer l'embedder
    sortie_embedder = [torch.stack(model(trajectoires)) for trajectoires in especes]

    # Sauvegarder la sortie
    sortie_chemin = os.path.join(output_dir, nom_fichier)
    torch.save(sortie_embedder, sortie_chemin)
    print(f"✅ Sauvegardé dans : {sortie_chemin}\n")



# Afficher tous les fichiers enregistrés
saved_files = sorted(f for f in os.listdir(output_dir) if f.endswith('.pt'))
print("\nles features obtenir à la sortie de l'embedder sont enregistrées dans  'featuress' :")
for f in saved_files:
    print(f"- {f}")
