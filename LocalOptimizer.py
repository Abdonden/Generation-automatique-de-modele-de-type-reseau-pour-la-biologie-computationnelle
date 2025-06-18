import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, global_mean_pool, global_max_pool

from embedding import MLP
from data import *
from Blocks import *




class Encoder (nn.Module):
    def __init__(self, n_species, d, dropout=0.1, num_layers = 1, device=None):
        super().__init__()
        self.h_0 = nn.Parameter(torch.randn(n_species,d)).reshape(1,n_species,d).repeat(num_layers, 1, 1)
        if device != None:
            self.h_0 = self.h_0.to(device)
        self.mdl = nn.RNN(2, d, num_layers=num_layers, batch_first=True)

    def forward (self, x):
        n_points = x.size(0)

        return self.mdl(x, self.h_0)[0] # [n_species, n_timepoints, d]


class CausalGNN (nn.Module):
    def __init__(self, n_species, d, d_prime, A=None, device=None, edges=None):
        super().__init__()
        if A == None:
            self.A = nn.Parameter(torch.eye(n_species))
            #self.A = nn.Parameter(torch.randn([n_species, n_species]))
        else:
            self.A = A

        self.device = device
        if device == None:
            self.device = torch.device("cpu")

        self.n_species = n_species
        self.W = nn.Linear(d,d_prime)

        mask = 1-torch.eye(n_species)
        if edges != None:
            mask = torch.zeros([n_species, n_species])
            src, dst = edges[0,:], edges[1,:]
            mask[src,dst]=1
            self.edges = edges
            for i in range (n_species):
                mask[i,i] = 0 #self loops are added manually

        self.register_buffer('mask', mask)


    def forward (self, x):

        mat = self.getMatrix()
        x = torch.matmul(mat, x)
        #x = torch.matmul(self.A, x)
        x = self.W(x)
        #x = F.relu(x)
        return x

    def getMatrix(self):
        mat = F.sigmoid(self.A)*self.mask
        return mat + torch.eye(self.n_species, device=self.device)

    def getEdges(self):
        mat = self.getMatrix()
        src, dst = self.edges[0,:], self.edges[1,:]
        return mat[src, dst]


class CausalityInference (nn.Module):
    def __init__(self, n_species,
                 f_theta, # RNN encoder
                 g_mu, g_mu_prime, # decoder causal et decoder standard
                 d, d_prime, # embedding dimensions
                 edges = None,
                 device=None
                 ):
        super().__init__()
        self.n_species = n_species
        self.encoder = f_theta
        self.decoder_causal = g_mu
        self.decoder_std = g_mu_prime
        self.gnn = CausalGNN(n_species, d, d_prime, edges=edges, device=device)
#        self.A = CausalGNN(n_species, d, d_prime).A
#        self.gnn = nn.Sequential (
#                        CausalGNN(n_species, d, d_prime, A=self.A),
#                        nn.ReLU(),
#                        CausalGNN(n_species, d_prime, d_prime, A=self.A))


        # pour l'apprentissage sans causal
        self.mlps = nn.ModuleList([
                        ResNetBlock(d, d_prime*2, d_prime*2, dropout=0) for i in range(n_species)
                    ])

    def forward(self,
                times,
                x): #[n_species, n_timepoints]
        n_species, n_timepoints = x.size()
        x = torch.stack([x,times.transpose(1,2).squeeze()], dim=-1) #ajout du temps
        x = self.encoder(x) # [n_species, n_timepoints, d] : embeddings des timepoints de chaque serie
        x = x.transpose(0,1) # [n_timepoints,  n_species, d] : feature matrix at each timepoint

        #translate the embeddings and use H^0 as the first embedding
        x = x[1:] 
        x = torch.cat((self.encoder.h_0[0].unsqueeze(0), x), dim=0)

        out_causal = self.gnn(x) # [n_timepoints, n_species,d']
        graph_embedding = out_causal.mean(dim=1).unsqueeze(1).repeat(1,n_species,1) 
        out_causal = torch.cat((out_causal, graph_embedding), dim=-1)
        out_causal = self.decoder_causal(out_causal)
        out_causal = out_causal.transpose(0,1)
        #out_causal = self.decoder_causal(out_causal) # [n_timepoints, n_species, 1]
        #out_causal = out_causal.transpose(0,1) # [n_species, n_timepoints]

        out_std = x.transpose(0,1) # [n_species, n_timepoints, d]
        out_std = [mlp(out_std[i]) for i, mlp in enumerate(self.mlps)] #[n_species,n_timepoints, d']
        out_std = torch.stack(out_std) 

        out_std = self.decoder_std(out_std)



        out_std = out_std.squeeze()
        out_causal = out_causal.squeeze()

        return out_causal, out_std









