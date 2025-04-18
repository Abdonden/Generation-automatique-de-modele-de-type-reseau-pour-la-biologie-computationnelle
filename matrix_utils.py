import torch
import libsbml
import networkx as nx
from torch_geometric.utils import to_dense_adj, add_self_loops
import re
import os

def pad_model_id(model_id):
    """Formate l'ID sur 4 chiffres pour un nom de fichier homogène."""
    return f"{int(model_id):04d}"

def generer_matrix(nom_fichier, model_id):
    model_id_str = pad_model_id(model_id)
    print(f"\n📂 Traitement du fichier : {nom_fichier} (ID: {model_id_str})")

    document = libsbml.readSBML(nom_fichier)
    model = document.getModel()

    if model is None:
        print(f"❌ Erreur : Impossible de lire le modèle SBML : {nom_fichier}")
        return

    species = {s.getId(): i for i, s in enumerate(model.getListOfSpecies())}

    ordered_nodes = [s for s, idx in sorted(species.items(), key=lambda x: x[1])]
    G = nx.Graph()
    G.add_nodes_from(ordered_nodes)

    interactions = []
    connected_species = set()

    for reaction in model.getListOfReactions():
        reactants = [r.getSpecies() for r in reaction.getListOfReactants()]
        products = [p.getSpecies() for p in reaction.getListOfProducts()]
        for r in reactants:
            for p in products:
                interactions.append([r, p])
                G.add_edge(r, p)
                connected_species.update([r, p])

    # Ajouter les espèces isolées
    isolated_species = set(species.keys()) - connected_species
    for s in isolated_species:
        G.add_node(s)

    mapping = {node: i for i, node in enumerate(ordered_nodes)}
    edge_list = []
    for source, target in G.edges():
        edge_list.append([mapping[source], mapping[target]])
        edge_list.append([mapping[target], mapping[source]])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = add_self_loops(edge_index)[0]

    adj_matrix = to_dense_adj(edge_index, max_num_nodes=len(species)).squeeze()

    interaction_indices = []
    for source, target in interactions:
        idx1 = species[source]
        idx2 = species[target]
        pair = [str(idx1), str(idx2)]
        interaction_indices.append(pair)

    # Création du dossier si besoin
    os.makedirs("adj", exist_ok=True)
    os.makedirs("interactions", exist_ok=True)
    os.makedirs("edges", exist_ok=True)

    # Sauvegarde
    torch.save(adj_matrix, f"adj/adj_matrix_{model_id_str}.pt")
    torch.save(edge_index, f"edges/edge_index_{model_id_str}.pt")
    torch.save(interaction_indices, f"interactions/interaction_indices_{model_id_str}.pt")

    print(f"✅ Fichiers enregistrés pour modèle {model_id_str} :")
    print(f" - adj/adj_matrix_{model_id_str}.pt")
    print(f" - edges/edge_index_{model_id_str}.pt")
    print(f" - interactions/interaction_indices_{model_id_str}.pt")


# Liste des fichiers SBML à traiter
fichiers = [
    "biomodels/BIOMD0000000001.xml",
    "biomodels/BIOMD0000000002.xml",
    "biomodels/BIOMD0000000082.xml",
    "biomodels/BIOMD0000000004.xml",
    "biomodels/BIOMD0000000086.xml",
    "biomodels/BIOMD0000000085.xml",
    "biomodels/BIOMD0000000014.xml",
    "biomodels/BIOMD0000000008.xml",
    "biomodels/BIOMD0000000011.xml",
    "biomodels/BIOMD0000000010.xml"
]

# Traitement de chaque fichier
for fichier in fichiers:
    match = re.search(r'BIOMD0*(\d+)', fichier)
    if match:
        model_id = int(match.group(1))  # Extrait 1, 85, etc.
        generer_matrix(fichier, model_id)
    else:
        print(f"⚠️ Impossible d'extraire l'ID du modèle depuis : {fichier}")
