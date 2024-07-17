import os.path as osp
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch_geometric.nn import to_hetero   
from torch_geometric.nn import global_mean_pool 

from tqdm import tqdm

hetero = True

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        #print("Data: ", data)
        #print("Data x_dict: ", data.x_dict)
        #print("Data batch dict: ", data.batch_dict)
        optimizer.zero_grad()
        output = model(data)
        if hetero:
            output = sum(global_mean_pool(data.x_dict[v], data.batch_dict[v]) for v in data.node_types)
        #print("Output: ", output.size)
        loss = F.nll_loss(output, data.y)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()


# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        if hetero:
            output = sum(global_mean_pool(data.x_dict[v], data.batch_dict[v]) for v in data.node_types)
        #print("Output: ", output)
        pred = output.max(dim=1)[1]
        #print("Pred: ", pred)
        #print("Data.y: ", data.y)
        correct += pred.eq(data.y).sum().item()
        #print(pred.eq(data.y))
    return correct / len(loader.dataset)


# 10-CV for GNN training and hyperparameter selection.
def gnn_evaluation(gnn, dataset, layers, hidden, max_num_epochs=200, batch_size=64, start_lr=0.01, min_lr = 0.000001, factor=0.5, patience=5,
                       num_repetitions=10, all_std=True):
    

    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []

    for i in range(num_repetitions):
        print("Repetition: ", i)
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)
        dataset.shuffle()

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0

            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                print("Layer: ", l)
                for h in hidden:
                    print("Hidden: ", h)
                    # Setup model.
                    model = gnn(dataset, l, h)
                    model.to(device)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    for epoch in tqdm(range(1, max_num_epochs + 1)):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        #print("train_loader: ", train_loader)
                        #print(len(train_loader.dataset))
                        train(train_loader, model, optimizer, device)
                        val_acc = test(val_loader, model, device)
                        print("Val acc: ", val_acc)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test = test(test_loader, model, device) * 100.0

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break
            
            test_accuracies.append(best_test)

            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    # print("All accuracies length: ", len(test_accuracies_all))
    # print("Complete accuracies length: ", len(test_accuracies_complete))

    if all_std:
        return (np.array(test_accuracies_all).mean(),
                np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(),
                np.array(test_accuracies_all).std())




def gnn_evaluation_wandb(gnn, dataset, layers, hidden, max_num_epochs=200, batch_size=128, start_lr=0.01, min_lr = 0.000001, factor=0.5, patience=5,
                       num_repetitions=10, all_std=True):
    
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'layers': {
                'values': layers
            },
            'hidden_dims': {
                'values': hidden
            }
        }
    }
    
    # sweep_id = wandb.sweep(sweep_config, project="gnn_evaluation")
    # wandb.init(project="gnn_evaluation")


    # Set device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_accuracies_all = []
    test_accuracies_complete = []

    for i in range(num_repetitions):
        print("Repetition: ", i)
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)
        dataset.shuffle()

        for train_index, test_index in kf.split(list(range(len(dataset)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_test = 0.0

            # Split data.
            train_dataset = dataset[train_index.tolist()]
            val_dataset = dataset[val_index.tolist()]
            test_dataset = dataset[test_index.tolist()]

            # Prepare batching.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

            # Collect val. and test acc. over all hyperparameter combinations.
            for l in layers:
                for h in hidden:
                    # Setup model.
                    model = gnn(dataset, l, h).to(device)
                    model.reset_parameters()

                    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                           factor=factor, patience=patience,
                                                                           min_lr=0.0000001)
                    for epoch in tqdm(range(1, max_num_epochs + 1)):
                        lr = scheduler.optimizer.param_groups[0]['lr']
                        train(train_loader, model, optimizer, device)
                        val_acc = test(val_loader, model, device)
                        scheduler.step(val_acc)

                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_test = test(test_loader, model, device) * 100.0
                        
                        #wandb.log({"val_acc": val_acc, "test_acc": best_test, "lr": lr})

                        # Break if learning rate is smaller 10**-6.
                        if lr < min_lr:
                            break
            
            test_accuracies.append(best_test)

            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    # print("All accuracies length: ", len(test_accuracies_all))
    # print("Complete accuracies length: ", len(test_accuracies_complete))
    
    if all_std:
        return (np.array(test_accuracies_all).mean(),
                np.array(test_accuracies_all).std(),
                np.array(test_accuracies_complete).std())
    else:
        return (np.array(test_accuracies_all).mean(),
                np.array(test_accuracies_all).std())



def train_wandb(gnn,dataset,max_num_epochs=200,start_lr=0.01, min_lr = 0.000001,batch_size=128,factor=0.5, patience=5):

    # wandb.init(project="gnn_evaluation")
    # configs = {
    #     'layers' : 1,
    #     'hidden_dims' : 32
    # }
    # wandb.config.update(configs)
    
    # Setup model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = gnn(dataset, wandb.config.layers, wandb.config.hidden_dims).to(device)
    #model.reset_parameters()

    dataset.shuffle()
    train_index, test_index = train_test_split(list(range(len(dataset))), test_size=0.1)
    train_index, val_index = train_test_split(train_index, test_size=0.1)

    # Split data.
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]

    # Prepare batching.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



    #optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                         factor=factor, patience=patience,
    #                                                         min_lr=0.0000001)
    #for epoch in tqdm(range(1, max_num_epochs + 1)):
        #lr = scheduler.optimizer.param_groups[0]['lr']
        #train(train_loader, model, optimizer, device)
        #val_acc = test(val_loader, model, device)
        #scheduler.step(val_acc)

        #test_acc = test(test_loader, model, device) * 100.0
        
        #wandb.log({"val_acc": val_acc, "test_acc": test_acc, "lr": lr, "epoch": epoch})

        # Break if learning rate is smaller 10**-6.
        # if lr < min_lr:
        #     break