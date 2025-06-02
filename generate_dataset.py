# generate_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import os
from data import *
import numpy as np
from scipy.interpolate import interp1d
from graph_utils import *
from tqdm import tqdm

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
        features = []
        for i in range(len(self)):
            try:
                features.append(self[i]["features"])
            except:
                pass
        total_sum = sum([feat.sum() for feat in features])
        nb_total = sum([torch.tensor(feat.shape).prod() for feat in features])
        self.mean = total_sum / nb_total

        variance = sum([((feat - self.mean) ** 2).sum() for feat in features]) / nb_total
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
            "features": features,
            "edges": edges,
            "interactions": interactions
        }

# Fonction pour obtenir les ID des modèles disponibles
def get_available_model_ids():
    model_ids = []
    for fname in os.listdir(FEATURES_DIR):
        if fname.startswith("features_") and fname.endswith(".pt"):
            model_id = fname.replace("features_", "").replace(".pt", "")
            model_id_str = str(model_id).zfill(4)

            edge_path = os.path.join(EDGE_DIR, f"edge_index_{model_id_str}.pt")
            inter_path = os.path.join(INTERACTIONS_DIR, f"interaction_indices_{model_id_str}.pt")
            feat_path = os.path.join(FEATURES_DIR, f"features_{model_id_str}.pt")

            if os.path.exists(edge_path) and os.path.exists(inter_path) and os.path.exists(feat_path):
                model_ids.append(int(model_id))
            else:
                print(f"⚠️ Fichiers manquants pour modèle {model_id_str}, ignoré.")
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


def valid_trajectories(feats):
    ret = []
    n_species, n_trajectories, n_points = feats.shape
    for i in range(n_trajectories):
        ok = True
        for s in range(n_species):
            if torch.any(torch.isnan(feats[s][i])):
                print("nan detected in traj", i, "of specie", s)
                ok = False
                raise ValueError("One trajectory rejected")
        if ok:
            ret.append(i)
    return torch.tensor(ret)

def logarithmic_resampling(traj):
    data = traj
    t = np.arange(1, len(data) + 1)
    t_log = np.logspace(np.log10(t[0]), np.log10(t[-1]), num=len(t), base=10.0)
    interp_func = interp1d(t, data, kind='linear', fill_value="extrapolate")
    data_log = interp_func(t_log)
    return torch.tensor(data_log)



def process_entry(entry):
        identifier = entry["model_id"]
        feats = entry["features"]
        edges = entry["edges"]
        target = entry["interactions"]

        try:
            traj_idx = valid_trajectories(feats)
        except Exception as e:
            print("skip model:", identifier)
            return None


        feats = feats[:, traj_idx, :]
        if len(traj_idx) == 1:
            feats = feats.permute(-1, 1, 0)


        n_target = []
        n_labels = []
        tot_0, tot_1 = 0, 0

        for j in range(len(target)):
            src, dst = target[j][0], target[j][1]
            n_edge = torch.tensor([src, dst])
            inverse = torch.tensor([dst, src])

            if not (check_if_edge_exists(inverse, target)):
                n_target.append(torch.tensor([dst, src]))
                n_labels.append(torch.tensor(0))
                n_target.append(n_edge)
                n_labels.append(torch.tensor(1))
                tot_0 += 1
                tot_1 += 1
            else:  # ANTISYM
                n_target.append(n_edge)
                n_labels.append(torch.tensor(1))
                tot_1 += 1

        if n_target == []:
            print("ERROR: empty for i=", i, " target=", target, " model=", identifier)
            return None
        else:
            n_target = torch.stack(n_target)
            n_labels = torch.stack(n_labels)
            data = MyData(x=feats, edge_index=edges, y=n_target, labels=n_labels)
            return data, tot_0, tot_1



if __name__ == "__main__":
    # Récupérer les modèles disponibles (ceux avec les trois fichiers existants)
    model_ids = get_available_model_ids()
    print(f" Modèles trouvés : {[str(mid).zfill(4) for mid in model_ids]}")

    triplets = lister_les_triplets(model_ids)
    print("\n Liste des triplets disponibles :")
    for triplet in triplets:
        print(f" - {triplet}")

    # Création du Dataset
    dataset = CRNDataset(model_ids)
    # dataset.compute_statistics()

    datas = []
    custom_dataset  = []
    tot_0, tot_1 = 0, 0

    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        ret = process_entry (entry)
        if ret is None:
            continue
        data, n_0, n_1 = ret
        tot_0 += n_0
        tot_1 += n_1
        datas.append(data)
        
        #subgraphs:
        sub_datas, n_0, n_1 = extract_all_subgraphs(data, entry["interactions"])
        custom_dataset = custom_dataset + sub_datas
        tot_0 += n_0
        tot_1 += n_1




    torch.save(datas, "dataset_sat.pt")
    torch.save(custom_dataset, "sub_dataset_sat.pt")
    print("\n✅ Dataset sauvegardé dans 'dataset_sat.pt'")
    print("Total size =", len(datas))
    print("mean =", dataset.mean, " std =", dataset.std)
    print("tot_0 =", tot_0, " tot_1 =", tot_1)
