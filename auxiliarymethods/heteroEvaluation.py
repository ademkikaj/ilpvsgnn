import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import Linear, SAGEConv, to_hetero
import torch
from torch_geometric.loader import DataLoader

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from datasets.Bongard.BongardDataset import BongardDataset

class GNNHetero(torch.nn.Module):
    def __init__(self,hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
def train(model,loader,optimizer):
    model.train()
    losses = 0
    for data in loader:  
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict) 
        x_dict = {key:global_mean_pool(data[key].x,data[key].batch) for key in data.node_types}
        x_values = list(x_dict.values())
        for i in range(len(x_values)):
            x_values[i] = x_values[i].sum(dim=1)
        
        out = torch.stack(x_values,dim=0).sum(dim=0)
        loss = F.nll_loss(out, data.y) 
        losses += loss.item()
        loss.requires_grad = True
        loss.backward()  
        optimizer.step() 
    return losses
    
    

def test(model,loader):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            pred = model(data.x_dict, data.edge_index_dict)
            pred = sum(global_mean_pool(data.x_dict[v],data.batch_dict[v]) for v in data.x_dict.keys())
        pred = pred.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    
    return correct / len(loader.dataset)

        

dataset = BongardDataset(root='datasets/Bongard/Heterogeneous', transform=T.ToUndirected())


train_dataset = dataset[:150]
val_dataset = dataset[150:170]
test_dataset = dataset[170:]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

data = train_dataset[0]
print(data)

print(data.metadata())
model = to_hetero(GNNHetero(32), data.metadata(), aggr='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(1, 200):
    losses = train(model,train_loader,optimizer)
    train_acc = test(model,train_loader)
    val_acc = test(model,val_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Loss: {losses:.4f}')