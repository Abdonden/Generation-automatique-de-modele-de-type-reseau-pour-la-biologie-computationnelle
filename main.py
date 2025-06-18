import torch

from data import *
from LocalOptimizer import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda

x = torch.load("dataset_small.pt")

def compute_grad(model, display=False, params_list=None):
        total_grad_norm = 0.0
        if params_list == None:
            params = model.named_parameters()
        else:
            params = params_list

        if params_list == None:
            for name, param in params:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    if display:
                        print(f" - {name} grad norm: {grad_norm:.6f}")
        else:
            for param in params:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
        if display:
            print(f" - Total gradient norm: {total_grad_norm:.6f}")
        return total_grad_norm



def fit_standard_loss(x, mdl, eps_tol=1e-2, max_epoch=100000, lr=1e-3, writer=None):
    #optimizes only the std loss
    optimizer = torch.optim.AdamW(#mdl.parameters(),
                                  list(mdl.encoder.parameters())+
                                  list(mdl.decoder_std.parameters())+
                                  list(mdl.mlps.parameters()),
                                  lr=lr)

    for i in range (max_epoch):
        optimizer.zero_grad()

        mdl.train()
        pred_causal, pred_std = mdl(x)

        loss = F.mse_loss(pred_std, x)
        loss.backward()

        optimizer.step()

        loss = loss.detach().item()

        if writer != None:
            writer.add_scalar("epoch/phase1_loss", loss, i)


        grad=compute_grad(mdl) #stoping condition on grad
        if i % 50 == 0:
            print ("[Phase 1] loss=",loss, " grad=", grad)
        if grad < eps_tol:
            return mdl, loss




def compute_nb_params(param_list):
    siz = 0
    for t in param_list:
        siz += torch.prod(torch.tensor(t.shape))
    return siz



def fit(data, d, d_prime,
        eps_tol = 1e-2, # breakpoint
        delta_min = 1e-3, # threshold where the causal loss is triggered
        alpha = 1e-2, #coefficient for the standard loss
        num_encoder_layers=1):
    x = data.x.squeeze()
    edges = data.y
    times = data.times.to(device)
    labels = data.labels.float().to(device)
    srcs = edges.transpose(0,1)[0,:]
    dsts = edges.transpose(0,1)[1,:]
    writer = SummaryWriter(comment="-1")

    n_species = x.size(0)
    encoder = Encoder(n_species, d, num_layers=num_encoder_layers, device=device)
    decoder_causal = nn.Linear(d_prime*2,1)
    decoder_std = nn.Linear(d_prime*2,1)
    mdl = CausalityInference(n_species, encoder, decoder_causal, decoder_std, d, d_prime,
                             edges = edges.transpose(0,1),
                             device=device)

    mdl, x = mdl.to(device), x.to(device)

    params_causal = list(mdl.gnn.parameters()) + list(mdl.decoder_causal.parameters())
#    params_std = list (mdl.mlps.parameters()) + list (mdl.decoder_std.parameters())
#
#
#    print ("nb_params causal=",compute_nb_params(params_causal), " std=", compute_nb_params(params_std))
    print ("nb_edges = ", len(srcs))
#    mdl, delta = fit_standard_loss(x, mdl, eps_tol=delta_min, writer=writer)
    delta = delta_min

    

    optimizer = torch.optim.Adam(mdl.parameters(),
                                  #list(mdl.gnn.parameters()) + list(mdl.decoder_causal.parameters()),
                                 #params_causal,
                                 lr=1e-4)

    print(mdl.gnn.getEdges())
    for i in range(1000000000):
        optimizer.zero_grad()

        mdl.train()
        pred_causal, pred_std = mdl(times,x)

        loss_causal = F.mse_loss(pred_causal, x)
        loss_std = F.mse_loss(pred_std, x)


        penalty = torch.trace(torch.matrix_exp(mdl.gnn.A)) - mdl.gnn.A.shape[0]
        #loss = loss_causal + alpha*F.relu(loss_std-delta)
        #loss =  loss_causal #+ mdl.gnn.A.norm(p=2)
        loss = loss_causal + alpha*penalty

        loss.backward()

        optimizer.step()

        loss_causal = loss_causal.detach().item()
        loss_std = loss_std.detach().item()
        loss = loss.detach().item()

        writer.add_scalar("epoch/loss", loss, i)
        writer.add_scalar("epoch/loss_causal", loss_causal, i)
        #writer.add_scalar("epoch/loss_std", loss_std, i)

        grad=compute_grad(mdl)
        writer.add_scalar("epoch/phase2_grad", grad,i)

        mat = mdl.gnn.getMatrix()
        coefs = mat[srcs,dsts]

        prediction_error = F.binary_cross_entropy(coefs, labels).detach().item()
        writer.add_scalar("epoch/phase2_error", prediction_error,i)

        if (i%50 == 0):
            #print ("[Phase2] pred_error: ", prediction_error, " causal=", loss_causal, " grad=", grad)
            #print ("[Phase2] Loss: ", loss, " causal=", loss_causal, " standard=", loss_std, " grad=", grad)
            print ("[Phase2] Loss: ", prediction_error, " causal=", loss_causal, " standard=", loss_std, " grad=", grad)
        if grad < eps_tol:
            print ("[Done] Loss: ", loss, " causal=", loss_causal, " grad=", grad)
            return mdl




