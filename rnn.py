#rnn originale
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset
from generate_dataset import MyData
from temporal_prediction import *
from Models import *


device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda
#device = torch.device("cpu")

writer = SummaryWriter(comment="")

def compute_grad(model):
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f" - {name} grad norm: {grad_norm:.6f}")
        print(f" - Total gradient norm: {total_grad_norm:.6f}")



def test_model(model, dloader):
    model.eval()
    n_examples = 0
    tot_loss = 0
    with torch.no_grad():
        for batch in dloader:
            feats, edges = batch.x, batch.edge_index
            feats, edges = feats.to(device).float(), edges.to(device)

            interaction_indices= batch.y.to(device)
            n_batched_examples = interaction_indices.shape[0]
            
            
            labels = batch.labels.to(device)
            labels = labels.long()

            src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination

            b = batch.batch.to(device)

            n_emb, (predxy,predyx), _ = model (b,feats,edges, src_idx, dst_idx)



            loss = criterion(predxy.squeeze() , labels.float())
            tot_loss += loss.detach()*n_batched_examples
            n_examples += n_batched_examples
    return tot_loss/n_examples
            


def train_model(model, batch):
    model.train()
    optimizer.zero_grad()
    batch = batch.to(device)
    feats, edges = batch.x, batch.edge_index
    feats, edges = feats.to(device).float(), edges.to(device)

    interaction_indices= batch.y.to(device)
    n_batched_examples = interaction_indices.shape[0]
    
    
    labels = batch.labels.to(device)
    labels = labels.long()

    src_idx, dst_idx = interaction_indices[:,0], interaction_indices[:,1] #indices des sommets source et destination

    b = batch.batch.to(device)

    n_embeddings, (predxy, predyx), _ = model (b,feats,edges, src_idx, dst_idx)


    n_positives = labels.sum()
    n_negatives = len(labels)-n_positives
    weights = (n_negatives/n_positives).to(device)



    # dir_loss: prediction de la direction des arcs (main task)
    # ep_loss : prediction de la présence d'un arc
    # antisym_loss: penalité sur l'antisymétrie
    #temporal_loss: penalité de reconstruction sur les séries temporelles TODO
    dir_loss = criterion(predxy.squeeze() , labels.float())
    ep_loss = evaluate_edge_prediction_loss(model, ep_model, batch, mask_ratio=0.1)
    temporal_loss = evaluate_timeseries_reconstruction(n_embeddings, feats, tmp_model)

    antisym_loss = ((F.sigmoid(predxy) + F.sigmoid(predyx)-1)**2).mean()
    
    main_loss = dir_loss + ep_loss + alpha*antisym_loss + temporal_loss

    main_loss.backward()
    old_lr = optimizer.param_groups[0]['lr']
    #scheduler.step(loss)
    new_lr = optimizer.param_groups[0]['lr']
    if old_lr != new_lr:
        print (f"[learning rate]: {old_lr} -> {new_lr}")
    optimizer.step()
    dir_loss = dir_loss.item()
    total_loss = dir_loss*n_batched_examples
    n_examples = n_batched_examples
    writer.add_scalar("batch/dir_loss_train", dir_loss, n)
    writer.add_scalar("batch/ep_loss_train",ep_loss, n)
    writer.add_scalar("batch/tmp_loss_train",temporal_loss, n)
    writer.add_scalar("batch/train",main_loss, n)
    return total_loss, n_examples





# === Chargement du dataset ===
src = 0
dst = src+1
standard_dataset = torch.load("dataset_sat.pt")
sub_dataset = torch.load("sub_dataset_sat.pt") #subgraphs

dataset = standard_dataset + sub_dataset

len_standard = len(standard_dataset)
print (int(len_standard*0.9))
standard_trn, standard_tst = standard_dataset[:int(len_standard*0.9)], standard_dataset[int(len_standard*0.9):]


trainset = standard_trn + sub_dataset
testset = standard_tst


#n_datas = len(dataset)
#trainset = dataset[:int(n_datas*0.9)]
#testset = dataset[int(n_datas*0.9):]
batch_size = 50 #len(trainset)
print ("batch_size=",batch_size)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=len(testset))

#examples statistics
nb_positive = torch.tensor(22517)
nb_negative = 20924
pos_weight = nb_negative/nb_positive

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# === Instanciation du modèle et optimiseur ===
in_channels = 512*3
out_channels=16
model_gcn = GCN(nb_points=in_channels, out_channels = out_channels, dropout=0.4)
model_gcn.to(device)

model = model_gcn
ep_model = MLP(out_channels*3, 1,out_channels*3, dropout=0).to(device).float()
tmp_model= MLP(out_channels, 10*in_channels,out_channels, dropout=0).to(device).float()

optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(ep_model.parameters()) + list(tmp_model.parameters()),
        lr=1e-3,weight_decay=1e-5)
#scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=100) 
n=0
best=10000
best_test = torch.tensor(10000)

alpha = 0.1 #antisymetry penalty coef

for i,epoch in enumerate(range(50000000000)):
    model.train()

    total_loss = 0.0
    n_examples = 0
    for j,batch in enumerate(dataloader):
        n+=1
        b_loss, b_size =train_model(model, batch) #on real_batch
        #compute_grad(model)
        total_loss += b_loss
        n_examples += b_size




    avg_loss = total_loss / n_examples
    if avg_loss < best:
        best=avg_loss
    writer.add_scalar("epoch/train", avg_loss, i)

    test_loss = test_model(model, testloader)
    writer.add_scalar("epoch/test", test_loss, i)
    if best_test > test_loss:
        best_test = test_loss


    # Logs
    if epoch % 500 == 0:
        print(f"\n🔧 Norme des gradients à l'époque {epoch} :")
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                print(f" - {name} grad norm: {grad_norm:.6f}")
        print(f" - Total gradient norm: {total_grad_norm:.6f}")

    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f} test={test_loss:.4f} best={best} best_test={best_test}")
    #print(f"Epoch {epoch} - Perte moyenne : {avg_loss:.4f} Test:{test_loss:.4f}")







# === Perte après entraînement ===
loss_after = evaluate_model(model_gcn, batch_data, interaction_indices_batch, criterion)
print(f"\n📊 Perte moyenne après entraînement : {loss_after:.4f}")


