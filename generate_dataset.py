# generate_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import os

# Répertoires où se trouvent les fichiers de données
FEATURES_DIR = "featuress"
EDGE_DIR = "edges"
INTERACTIONS_DIR = "interactions"

# Définition d'une classe Dataset personnalisée
class CRNDataset(Dataset):
    # __init__(self, model_ids) : Le constructeur de la classe prend en argument une liste de model_ids (identifiants des modèles).
    def __init__(self, model_ids):
        self.model_ids = model_ids

    # __len__(self) : Cette méthode retourne la taille de l'ensemble de données, ici le nombre de modèles (i.e., la longueur de model_ids).
    def __len__(self):
        return len(self.model_ids)

    # __getitem__(self, idx) : Cette méthode est appelée pour récupérer un élément de l'ensemble de données à un indice donné.
    # L'argument idx est l'indice du modèle que l'on souhaite récupérer.
    def __getitem__(self, idx):
        # model_id = self.model_ids[idx] : Le modèle est récupéré à l'indice idx dans la liste des model_ids.
        model_id = self.model_ids[idx]
        # model_id_str = str(model_id).zfill(4) : L'identifiant du modèle est converti en chaîne de caractères,
        # et zfill(4) permet d'ajouter des zéros devant si nécessaire pour avoir une chaîne de 4 caractères (par exemple, 4 devient 0004).
        model_id_str = str(model_id).zfill(4)

        # Ces lignes construisent les chemins des fichiers .pt (fichiers PyTorch) pour les caractéristiques, la matrice d'adjacence et les indices d'interactions. 
        features_path = os.path.join(FEATURES_DIR, f"features_{model_id_str}.pt")
        edges_path = os.path.join(EDGE_DIR, f"edge_index_{model_id_str}.pt")
        interactions_path = os.path.join(INTERACTIONS_DIR, f"interaction_indices_{model_id_str}.pt")

        # torch.load : Charge les fichiers .pt correspondants aux caractéristiques, à la matrice d'adjacence et aux interactions.
        # L'argument weights_only=True signifie que seuls les poids du modèle sont chargés, et non d'autres objets PyTorch inutiles
        features = torch.load(features_path, weights_only=True)
        edges = torch.load(edges_path, weights_only=True)
        interactions = torch.load(interactions_path, weights_only=True)

        return {
            "model_id": model_id_str,
            "features": features,
            "edges": edges,
            "interactions": interactions
        }

# Fonction pour obtenir les ID des modèles disponibles
# get_available_model_ids : Cette fonction parcourt tous les fichiers du répertoire FEATURES_DIR pour trouver ceux qui commencent par features_ et se terminent par .pt.
# Ensuite, elle extrait l'ID du modèle à partir du nom du fichier, le convertit en entier et l'ajoute à la liste model_ids.
# sorted(model_ids) : La fonction retourne cette liste triée des identifiants de modèles disponibles.
def get_available_model_ids():
    model_ids = []
    for fname in os.listdir(FEATURES_DIR):
        if fname.startswith("features_") and fname.endswith(".pt"):
            model_id = fname.replace("features_", "").replace(".pt", "")
            model_ids.append(int(model_id))
    return sorted(model_ids)

# Fonction pour lister les triplets de fichiers
# lister_les_triplets(model_ids) : Cette fonction prend une liste d'IDs de modèles et crée une liste de triplets
# (chemin des fichiers de features, adjacence, et interactions pour chaque modèle)
def lister_les_triplets(model_ids):
    triplets = []
    for mid in model_ids:
        mid_str = str(mid).zfill(4)
        triplet = (
            f"{FEATURES_DIR}/features_{mid_str}.pt",
            f"{EDGE_DIR}/edge_index_{mid_str}.pt",
            f"{INTERACTIONS_DIR}/interaction_indices_{mid_str}.pt"
        )
        triplets.append(triplet)
    return triplets

# Fonction collate personnalisée pour regrouper tous les éléments du dataset en un seul batch
# Ici on retourne un dictionnaire avec des listes de chaque type d'élément (features, adj_matrix, etc.)
def custom_collate(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [sample[key] for sample in batch]
    return batch_dict

if __name__ == "__main__":
    # Récupérer tous les modèles automatiquement
    model_ids = get_available_model_ids()  # <- ça charge tous les modèles disponibles automatiquement

    print(f" Modèles trouvés : {[str(mid).zfill(4) for mid in model_ids]}")

    triplets = lister_les_triplets(model_ids)
    print("\n Liste des triplets disponibles :")
    for triplet in triplets:
        print(f" - {triplet}")

    # Création du Dataset et DataLoader avec le collate personnalisé
    dataset = CRNDataset(model_ids)
    print(len(dataset))
    #dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=custom_collate)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=custom_collate)
    # On récupère le seul batch du DataLoader (contenant tout le dataset)
    batch = next(iter(dataloader))

    print("\n Contenu complet du dataset chargé en batch :")
    for i in range(len(batch["model_id"])):
        print(f"\n Modèle {i+1}")
        print(f" - Model ID: {batch['model_id'][i]}")
        #print(f"   Features shape: {batch['features'][i].shape}")
        print("   Features shapes (par espèce) :")
        for j, esp in enumerate(batch["features"][i]):
            print(f"     - Espèce {j} : {esp.shape}")
        print(f"   edges shape: {batch['edges'][i].shape}")
        print(f"   # Interactions: {len(batch['interactions'][i])}")
        # Tu peux décommenter la ligne suivante si tu veux voir les interactions elles-mêmes
        print(f"   Interactions: {batch['interactions'][i]}")  

    #  Sauvegarde du batch complet dans un fichier .pt pour usage ultérieur
    torch.save(batch, "dataset_complet.pt")
    print("\n Batch complet sauvegardé dans 'dataset_complet.pt'")
