import os
import re
import torch
from torch.utils.data import Dataset

# Import des fonctions
from generer_donnees import generer_donnees
from matrix_utils import generer_matrix


# Paramètres
tmax = 10
biomodels = [
    "biomodels/BIOMD0000000001.xml",
    "biomodels/BIOMD0000000002.xml",
    "biomodels/BIOMD0000000003.xml",
    "biomodels/BIOMD0000000004.xml",
    "biomodels/BIOMD0000000005.xml",
    "biomodels/BIOMD0000000085.xml",
    "biomodels/BIOMD0000000007.xml",
    "biomodels/BIOMD0000000008.xml",
    "biomodels/BIOMD0000000009.xml",
    "biomodels/BIOMD0000000010.xml"
]

# Dossier de sortie
os.makedirs("features", exist_ok=True)

# Étape 1 : Génération des .pt
print("🚀 Génération des fichiers .pt")
for fichier in biomodels:
    try:
        print(f"\n🧪 Traitement du fichier : {fichier}")
        
        # Extraire l'ID du modèle
        match = re.search(r'BIOMD0*(\d+)', fichier)
        if not match:
            print(f"⚠️ ID introuvable dans : {fichier}")
            continue
        model_id = int(match.group(1))
        model_id_str = str(model_id)

        # Générer les données temporelles
        features = generer_donnees(fichier, tmax)
        torch.save(features, f"features/features_{model_id_str}.pt")

        # Générer les matrices associées
        generer_matrix(fichier, model_id)

    except Exception as e:
        print(f"❌ Erreur lors du traitement de {fichier} : {e}")

# Étape 2 : Chargement des triplets
print("\n📦 Chargement des triplets")
triplets = []
model_ids = []

for fname in os.listdir("features"):
    match = re.match(r"features_(\d+)\.pt", fname)
    if match:
        model_ids.append(int(match.group(1)))

model_ids.sort()

for model_id in model_ids:
    try:
        features = torch.load(f"features_{model_id}.pt")
        adj = torch.load(f"adj_matrix_{model_id}.pt")
        indices = torch.load(f"interaction_indices_{model_id}.pt")
        triplets.append((adj, indices, features))
    except FileNotFoundError as e:
        print(f"⚠️ Fichier manquant pour le modèle {model_id} : {e}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {model_id} : {e}")

# Étape 3 : Définir le Dataset PyTorch
class GraphDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

dataset = GraphDataset(triplets)
print(f"\n✅ Dataset prêt avec {len(dataset)} échantillons.")
