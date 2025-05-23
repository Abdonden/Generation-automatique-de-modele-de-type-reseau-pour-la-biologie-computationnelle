
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import torch.nn.functional as F
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



def evaluate_edge_prediction_loss(
                          gcn, #produces the node embeddings
                          head_model, #processes the node embedding to predict if there is an edge
                          batch, mask_ratio=0.2, negative_ratio=None):
    new_batch, query, labels = perturbate_batch(batch, mask_ratio, negative_ratio)

    src_idx, dst_idx = query[0,:], query[1,:]
    feats, edges = new_batch.x, new_batch.edge_index

    feats = feats.float()
    _, xy = gcn(new_batch.batch, feats, edges,src_idx, dst_idx)

    pred = head_model(xy)

    n_positives = labels.sum()
    n_negatives = len(labels)-n_positives
    weights = (n_negatives/n_positives)

    labels = labels.unsqueeze(-1)
    loss = F.binary_cross_entropy_with_logits(pred, labels, pos_weight= weights)
    return loss

