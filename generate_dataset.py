# generate_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import os
from data import *

# Répertoires où se trouvent les fichiers de données
FEATURES_DIR = "features"
EDGE_DIR = "edges"
INTERACTIONS_DIR = "interactions"

# Définition d'une classe Dataset personnalisée
class CRNDataset(Dataset):
    def __init__(self, model_ids):
        self.model_ids = model_ids
        self.mean = 0
        self.std = 1
        self.min, self.max = None, None

    def __len__(self):
        return len(self.model_ids)
    
    def compute_statistics(self):
        #features = [self[i]["features"] for i in range(len(self))]
        features = []
        for i in range (len(self)):
            try:
                features.append(self[i]["features"])
            except:
               pass 
        total_sum = sum([feat.sum() for feat in features])
        nb_total = sum([torch.tensor(feat.shape).prod() for feat in features])
        self.mean = total_sum / nb_total

        variance = sum([((feat - self.mean)**2).sum() for feat in features]) / nb_total
        self.std = torch.sqrt(variance)

        self.min = min([feat.min() for feat in features])
        self.max = max([feat.max() for feat in features])

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        model_id_str = str(model_id).zfill(4)

        features_path = os.path.join(FEATURES_DIR, f"features_{model_id_str}.pt")
        edges_path = os.path.join(EDGE_DIR, f"edge_index_{model_id_str}.pt")
        interactions_path = os.path.join(INTERACTIONS_DIR, f"interaction_indices_{model_id_str}.pt")

        features = torch.load(features_path, weights_only=True)

        #if (torch.any(features <= 0)):
        #    print ("error: negative features for instance ",i)
        #    raise ValueError

        #features = torch.log(features)
        edges = torch.load(edges_path, weights_only=True)
        interactions = torch.load(interactions_path, weights_only=True)
        interactions = torch.tensor([[int(i), int(j)] for i, j in interactions])

        eps = torch.tensor(1e-80)
        features = torch.max(features, eps)
        maximums = features.max(dim=-1)[0]
        maximums = maximums.unsqueeze(-1).expand(features.shape)
        features = features / maximums

        return {
            "model_id": model_id_str,
            #"features": (features - self.mean) / self.std,
            #"features":features,
            "features":features,
            "edges": edges,
            "interactions": interactions
        }

# Fonction pour obtenir les ID des modèles disponibles
def get_available_model_ids():
    model_ids = []
    for fname in os.listdir(FEATURES_DIR):
        if fname.startswith("features_") and fname.endswith(".pt"):
            model_id = fname.replace("features_", "").replace(".pt", "")
            model_ids.append(int(model_id))
    return sorted(model_ids)

# Fonction pour lister les triplets de fichiers
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

def check_if_edge_exists (edge, interaction_indices):
    """
        Vérifie si un arc est déjà présent dans la liste des indices (slow mais bon)
    """

    for i in range(len(interaction_indices)):
        e = interaction_indices[i]
        if e[0] == edge[0] and e[1] == edge[1]:
            return True
    return False

def valid_trajectories (feats):
    """
        identifies the set of indices where each trajectory is valid (no NaN)
    """
    ret = []
    n_species, n_trajectories, n_points = feats.shape
    for i in range (n_trajectories):
        ok = True
        for s in range(n_species):
            if torch.any(torch.isnan(feats[s][i])):
                print ("nan detected in traj ", i, " of specie ", s)
                ok=False
        if ok:
            ret.append(i)
    if ret == []:
        raise ValueError ("no valid trajectory for this model")
    return torch.tensor(ret)

   
if __name__ == "__main__":
    # Récupérer tous les modèles automatiquement
    model_ids = get_available_model_ids()
    print(f" Modèles trouvés : {[str(mid).zfill(4) for mid in model_ids]}")

    triplets = lister_les_triplets(model_ids)
    print("\n Liste des triplets disponibles :")
    for triplet in triplets:
        print(f" - {triplet}")

    # Création du Dataset
    dataset = CRNDataset(model_ids)
    #dataset.compute_statistics()
    #print(f"mean = {dataset.mean:.4f}, std = {dataset.std:.4f}")

    # Transformation en Data (PyG) et sauvegarde
    datas = []
    tot_0, tot_1 = 0,0

    max_nb_sommets = 0
    for i in range(len(dataset)):
        try:
            entry = dataset[i]
            feats = entry["features"]
            edges = entry["edges"]
            target = entry["interactions"]

            traj_idx = valid_trajectories(feats)
            feats = feats[:,traj_idx,:]


            if max_nb_sommets < len(feats):
                max_nb_sommets = len(feats)


            n_target = []
            n_labels = []
            for j in range(len(target)):
                src, dst = target[j][0], target[j][1]

                n_edge = torch.tensor([src,dst])

                inverse = torch.tensor([dst,src])
                #n_target.append(n_edge)
                if not(check_if_edge_exists(inverse, target)):
                   n_target.append(torch.tensor([dst,src]))
                   n_labels.append(torch.tensor(0))
                   n_target.append(n_edge)
                   n_labels.append(torch.tensor(1))
            
            n_target = torch.stack(n_target)
            n_labels = torch.stack(n_labels)



            #feats = (feats - feats.mean())/feats.std() 
            data = MyData(x=feats, edge_index=edges, y=n_target, labels=n_labels)
            datas.append(data)
        except Exception as e:
            print (f"[pass {i}]: {e}")
            continue

    torch.save(datas, "dataset_sat.pt")
    print("\n✅ Dataset sauvegardé dans 'dataset_sat.pt'")
    print ("Total size= ",len(datas))
    print ("Max graph size=",max_nb_sommets)
    print ("mean=",dataset.mean, " std=", dataset.std)
    print ("tot_0=", tot_0, " tot_1=", tot_1)

