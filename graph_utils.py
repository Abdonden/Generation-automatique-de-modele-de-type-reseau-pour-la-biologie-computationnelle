from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import negative_sampling
from data import *


def mask_random_edges(data, mask_ratio=0.2):
    edge_index = data.edge_index  # [2, E]
    num_edges = edge_index.size(1)

    # Tirage aléatoire des indices à conserver
    keep_mask = torch.rand(num_edges) > mask_ratio
    masked_edge_index = edge_index[:, keep_mask]

    # Optionnel : on peut aussi retourner les indices supprimés si on veut en faire une tâche SSL
    removed_edge_index = edge_index[:, ~keep_mask]

    # Clonage du data pour éviter de le modifier
    new_data = data.clone()
    new_data.edge_index = masked_edge_index


    return new_data, removed_edge_index


def generate_negative_edges (data, ratio=0.2):
    n_edges = data.edge_index.size(1)
    num_samples = None
    if ratio != None:
        num_samples = int(ratio*n_edges)
    ret = negative_sampling(
                edge_index = data.edge_index,
                num_nodes = data.num_nodes,
                num_neg_samples= num_samples,
                method="dense",
            )
    return ret



def perturbate_batch(batch, mask_ratio=0.2, negative_ratio=None):
    new_batch, removed_edges = mask_random_edges(batch, mask_ratio)
    negative_pairs = generate_negative_edges(batch, negative_ratio)

    query = torch.cat((removed_edges, negative_pairs), dim=1)
    ones = torch.ones(removed_edges.size(1), device=negative_pairs.device)
    zeros = torch.zeros(negative_pairs.size(1), device=negative_pairs.device)
    labels = torch.cat([ones, zeros])
    return new_batch, query, labels

def check_if_edge_exists(edge, interaction_indices):
    for i in range(len(interaction_indices)):
        e = interaction_indices[i]
        if e[0] == edge[0] and e[1] == edge[1]:
            return True
    return False

def is_isolated(node_idx, edge_index):
    return not ((edge_index[0] == node_idx).any() or (edge_index[1] == node_idx).any())



def extract_2hop_subgraph(node_idx, data,interaction_indices, num_hops=2):
    """
    Extrait le sous-graphe à num_hops sauts autour du noeud node_idx.

    Args:
        node_idx (int): index du noeud central (int)
        data (Data): graphe PyG (avec data.edge_index, data.x, etc.)
        num_hops (int): nombre de sauts (par défaut 2)

    """
    if is_isolated(node_idx, data.edge_index):
        return None
    # Obtenir les noeuds et arêtes du sous-graphe
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=data.edge_index,
        #relabel_nodes=True
    )
    # Créer un nouvel objet Data
    if edge_index.size(1) == 0:
        return None


    # Extraire les features associées
    sub_x = data.x #car on relabel pas
    #sub_x = data.x[subset]


    #recupération des arcs:
    ys, labels = [],[]

    n_0, n_1 = 0,0
    for edge_ in interaction_indices:
        src, dst = edge_
        edge = torch.stack([src,dst])
        if check_if_edge_exists(edge, edge_index):
            ys.append(edge)
            labels.append(1)
            inverse = torch.stack([dst, src])
            n_1 += 1

            if not check_if_edge_exists(inverse, interaction_indices):
                ys.append(inverse)
                labels.append(0)
                n_0 += 1
    
    if ys == []:
        return None #can be only self loops
    sub_y = torch.stack(ys)

    sub_labels = torch.tensor(labels)

    sub_data = MyData(x=sub_x, edge_index=edge_index, y=sub_y, labels=sub_labels)
    return sub_data, n_0, n_1

def extract_subgraph (data,interaction_indices, max_try=10):
    num_nodes = data.x.size(0)
    while max_try > 0:
        node_idx = torch.randint(0, num_nodes, (1,)).item()
        sub_data = extract_2hop_subgraph(node_idx, data, interaction_indices, 2)
        if not (sub_data is None):
            return sub_data
        max_try -= 1
    return None

def extract_all_subgraphs(data, interaction_indices, num_hops=2):
    num_nodes = data.x.size(0)
    ret = []
    tot_0, tot_1 = 0 ,0
    for i in range(num_nodes):
        sub_data_ = extract_2hop_subgraph(torch.tensor([i]), data, interaction_indices, ,num_hops=num_hops)
        if not (sub_data_ is None):
            sub_data, n_0, n_1 = sub_data_
            tot_0 += 1
            tot_1 += 1
            ret.append(sub_data)
    return ret, tot_0, tot_1


    


