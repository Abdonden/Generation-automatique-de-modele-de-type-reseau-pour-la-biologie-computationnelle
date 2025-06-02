


import torch.nn.functional as F
import torch



def evaluate_timeseries_reconstruction (embeddings, timeseries, tmp_model):
    pred = tmp_model(embeddings)
    pred = pred.reshape(timeseries.shape)
    return F.mse_loss(pred, timeseries)
