import torch                   # Pour la manipulation de tenseurs et sauvegarde avec torch.save
import libsbml                 # Pour lire les fichiers SBML (modèles de systèmes biologiques)
import networkx as nx          # Pour la manipulation de graphes
from torch_geometric.utils import to_dense_adj, add_self_loops  # Utilitaires pour les graphes en PyTorch Geometric
import re                      # Pour les expressions régulières (extraction d’ID de fichiers)
import os                      # Pour les opérations système (création de dossiers)

#Assure que tous les model_id soient formatés à 4 chiffres, par ex. 15 devient 0015
def pad_model_id(model_id):
    """Formate l'ID sur 4 chiffres pour un nom de fichier homogène."""
    return f"{int(model_id):04d}"

# Fonction principale de traitement de fichier SBML
def generer_matrix(nom_fichier, model_id):
# Affiche le fichier en cours de traitement.
    model_id_str = pad_model_id(model_id)
    print(f"\n📂 Traitement du fichier : {nom_fichier} (ID: {model_id_str})")
#Lit le fichier SBML et extrait le modèle
    document = libsbml.readSBML(nom_fichier)
    model = document.getModel()
#Vérifie si le modèle a bien été chargé.
    if model is None:
        print(f"❌ Erreur : Impossible de lire le modèle SBML : {nom_fichier}")
        return
#Crée un dictionnaire {id_espece: index}.
#model.getListOfSpecies():C'est une méthode de libSBML qui  donne la liste de toutes les espèces chimiques (molécules, protéines, etc.) dans le  modèle SBML.

# cette ligne crée un dictionnaire species dont :Clé = l'identifiant (string) de chaque espèce (par ex : "ATP", "glucose", etc.)
#                                               :Valeur = un numéro unique (son index dans la liste)
#enumerate(...)  donne un index i pour chaque espèce s   s.getId() donne son nom (ou ID)
    species = {s.getId(): i for i, s in enumerate(model.getListOfSpecies()) if not(s.getBoundaryCondition())}

# Trie les nœuds dans l'ordre de leurs indices.
    ordered_nodes = [s for s, idx in sorted(species.items(), key=lambda x: x[1])]
# Initialise un graphe NetworkX et y ajoute les nœuds.
    G = nx.Graph()
    G.add_nodes_from(ordered_nodes)
    print ("NODES=", ordered_nodes)
#interactions : stocke les paires (réactif, produit)
#connected_species : espèces connectées à au moins une autre.
    interactions = []
    connected_species = set()
#Pour chaque réaction : crée les liens entre réactifs et produits (r → p), ajoute les arêtes au graphe.
    for reaction in model.getListOfReactions():
        reactants = [r.getSpecies() for r in reaction.getListOfReactants()]
        products = [p.getSpecies() for p in reaction.getListOfProducts()]
        for r in reactants:
            for p in products:
                if r in species.keys() and p in species.keys(): 

                   # print ("AJOUT : ", r, " => ", p)
                    interactions.append([r, p])
                    G.add_edge(r, p)
                    connected_species.update([r, p])

#    # Ajouter les espèces isolées
#    isolated_species = set(species.keys()) - connected_species
#    for s in isolated_species:
#        if s in species.keys():
#            G.add_node(s)

#Donne un index à chaque nœud.
    mapping = {node: i for i, node in enumerate(ordered_nodes)}
   # print ("MAPPING=",mapping)
    edge_list = []
#Crée des arêtes bidirectionnelles (non orienté).
    for source, target in G.edges():
        edge_list.append([mapping[source], mapping[target]])
        edge_list.append([mapping[target], mapping[source]])
#cette ligne convertit edge_list en un tenseur PyTorch avec des entiers
#t() = transpose. Ça change les dimensions pour correspondre au format attendu par PyTorch Geometric 
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # boucles sur chaque nœud.
    edge_index = add_self_loops(edge_index)[0]
#Génère la matrice d’adjacence (dense) à partir des arêtes.
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=len(species)).squeeze()
#Stocke les paires d’espèces interagissantes sous forme d’indices.
    interaction_indices = []
    for source, target in interactions:
        idx1 = mapping[source]
        idx2 = mapping[target]
        pair = [str(idx1), str(idx2)]
        interaction_indices.append(pair)

    # Création du dossier 
    os.makedirs("adj", exist_ok=True)
    os.makedirs("interactions", exist_ok=True)
    os.makedirs("edges", exist_ok=True)

    # Sauvegarde
    torch.save(adj_matrix, f"adj/adj_matrix_{model_id_str}.pt")
    torch.save(edge_index, f"edges/edge_index_{model_id_str}.pt")
    torch.save(interaction_indices, f"interactions/interaction_indices_{model_id_str}.pt")
   #Affichage
    print(f"✅ Fichiers enregistrés pour modèle {model_id_str} :")
    print(f" - adj/adj_matrix_{model_id_str}.pt")
    print(f" - edges/edge_index_{model_id_str}.pt")
    print(f" - interactions/interaction_indices_{model_id_str}.pt")


# Liste automatique des 100 premiers fichiers SBML à traiter
#Extrait l’ID numérique avec regex.
#Appelle generer_matrix si l’ID est trouvé

# Dossier contenant les fichiers SBML
dossier_sbml = "biomodels"

# Récupère tous les fichiers .xml dans le dossier
tous_les_fichiers = [
    os.path.join(dossier_sbml, f)
    for f in os.listdir(dossier_sbml)
    if f.endswith(".xml") and f.startswith("BIOMD")
]

# Trie les fichiers par ID numérique extrait avec regex
fichiers_tries = sorted(
    tous_les_fichiers,
    key=lambda x: int(re.search(r'BIOMD0*(\d+)', x).group(1))
)

# Garde les 100 premiers fichiers
fichiers_a_traiter = fichiers_tries #fichiers_tries[:204]

# Traitement
for fichier in fichiers_a_traiter:
    match = re.search(r'BIOMD0*(\d+)', fichier)
    if match:
        model_id = int(match.group(1))  
        generer_matrix(fichier, model_id)
    else:
        print(f"⚠️ Impossible d'extraire l'ID du modèle depuis : {fichier}")

#print ("FINAL")
#generer_matrix("biomodels/BIOMD0000000007.xml",7)







 
  
