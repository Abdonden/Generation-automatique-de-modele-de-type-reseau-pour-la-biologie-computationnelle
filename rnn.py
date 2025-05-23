#rnn originale
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from generate_dataset import MyData
from Models import *


device = torch.device("cuda:0") #processeur = cpu, carte graphique = cuda
#device = torch.device("cpu")

writer = SummaryWriter(comment="")


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

            prediction, _ = model (b,feats,edges, src_idx, dst_idx)



            loss = criterion(prediction.squeeze() , labels.float())
            tot_loss += loss.detach()*n_batched_examples
            n_examples += n_batched_examples
    return tot_loss/n_examples
            



    


# === Chargement du dataset ===
src = 0
dst = src+1
dataset = torch.load("dataset_sat.pt")
n_datas = len(dataset)
trainset = dataset[:int(n_datas*0.9)]
testset = dataset[int(n_datas*0.9):]
batch_size = len(trainset)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=len(testset))

#examples statistics
nb_positive = torch.tensor(1854)
nb_negative = 1854 
pos_weight = nb_negative/nb_positive

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# === Instanciation du modèle et optimiseur ===
out_channels=16
model_gcn = GCN(out_channels = out_channels, dropout=0.1)
model_gcn.to(device)

model = model_gcn
head_model = MLP(out_channels*3, 1,out_channels*3, dropout=0.1).to(device).float()

optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head_model.parameters()),
        lr=1e-3,weight_decay=1e-5)
#scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=100) 
n=0
best=10000
best_test = torch.tensor(10000)

for i,epoch in enumerate(range(50000000000)):
    model.train()

    total_loss = 0.0
    n_examples = 0
    for j,batch in enumerate(dataloader):
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

        prediction, xy = model (b,feats,edges, src_idx, dst_idx)



        dir_loss = criterion(prediction.squeeze() , labels.float())
        ep_loss = evaluate_edge_prediction_loss(model, head_model, batch, mask_ratio=0.1)
        
        main_loss = dir_loss + ep_loss

        main_loss.backward()
        old_lr = optimizer.param_groups[0]['lr']
        #scheduler.step(loss)
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print (f"[learning rate]: {old_lr} -> {new_lr}")
        optimizer.step()
        dir_loss = dir_loss.item()
        total_loss += dir_loss*n_batched_examples
        n_examples += n_batched_examples
        writer.add_scalar("batch/dir_loss_train", dir_loss, n)
        writer.add_scalar("batch/ep_loss_train",ep_loss, n)
        n+=1

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
