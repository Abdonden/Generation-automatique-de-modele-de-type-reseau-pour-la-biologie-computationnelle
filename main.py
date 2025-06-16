import torch

from data import *
from LocalOptimizer import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




x = torch.load("dataset_small.pt")

def compute_grad(model):
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f" - {name} grad norm: {grad_norm:.6f}")
        print(f" - Total gradient norm: {total_grad_norm:.6f}")
        return total_grad_norm




def fit(x, d, d_prime, eps_tol = 1e-2, num_encoder_layers=1):
    x = x.x.squeeze()
    n_species = x.size(0)
    encoder = Encoder(n_species, d, num_layers=num_encoder_layers)
    decoder = nn.Linear(d_prime,1)
    mdl = CausalityInference(n_species, encoder, decoder, d, d_prime)

    optimizer = torch.optim.AdamW(mdl.parameters(),
                                  lr=1e-3)

    writer = SummaryWriter(comment="-1")
    for i in range(1000000000):
        optimizer.zero_grad()

        mdl.train()
        pred = mdl(x)

        loss = F.mse_loss(pred, x)
        loss.backward()

        optimizer.step()

        loss = loss.detach().item()

        writer.add_scalar("epoch/loss", loss, i)

        if (i%50 == 0):
            grad=compute_grad(mdl)
            writer.add_scalar("epoch/grad", grad,i)
            print ("Loss: ", loss)
        if loss < eps_tol:
            return mdl




