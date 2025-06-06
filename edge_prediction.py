
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import torch.nn.functional as F
from data import *
from graph_utils import *





def evaluate_edge_prediction_loss(
                          gcn, #produces the node embeddings
                          head_model, #processes the node embedding to predict if there is an edge
                          batch, mask_ratio=0.2, negative_ratio=None):
    new_batch, query, labels = perturbate_batch(batch, mask_ratio, negative_ratio)

    src_idx, dst_idx = query[0,:], query[1,:]
    feats, edges = new_batch.x, new_batch.edge_index

    feats = feats.float()
    n_emb, _, xy = gcn(new_batch, src_idx, dst_idx)

    pred = head_model(xy)

    n_positives = labels.sum()
    n_negatives = len(labels)-n_positives
    weights = (n_negatives/n_positives)

    labels = labels.unsqueeze(-1)
    loss = F.binary_cross_entropy_with_logits(pred, labels, pos_weight= weights)
    return loss

