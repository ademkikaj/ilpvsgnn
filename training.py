import wandb
from datasets.Bongard.BongardDataset import BongardDataset
from sklearn.model_selection import StratifiedShuffleSplit
from datasets.Muta.MutaDataset import MutaDataset
from torch_geometric.loader import DataLoader
from gnn_baselines.gnn_architectures import *
import torch
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
import numpy as np
import yaml
import math
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import KFold
from datasets.KRK.KRKDataset import KRKDataset
import numpy as np


def split_dataset(dataset,ratio):
    data_size = len(dataset)
    train = np.reshape(np.arange(data_size*2), (data_size, 2))
    test = np.random.randint(2, size=data_size)
    sss = ShuffleSplit(n_splits=1, test_size=ratio)
    sss.get_n_splits(train, test)
    train_indices, test_indices = next(sss.split(train, test))
    train = dataset[train_indices]
    test = dataset[test_indices]
    return train,test

def get_fold(datalist,fold):
    folds = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(datalist):
        folds.append((train_index, test_index))
    train_index, test_index = folds[fold]
    train = [datalist[i] for i in train_index.tolist()]
    test = [datalist[i] for i in test_index.tolist()]
    return train,test

# One training epoch for GNN model.
def train(train_loader, model, optimizer, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()

# Get acc. of GNN model.
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def train_wandb(config=None):

    # initialize a new wandb run
    
    # with open("wandb_config.yaml", 'r') as file:
    #     config = yaml.load(file, Loader=yaml.FullLoader)
    
    wandb.init()
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.dataset == "MUTAG":
        dataset = TUDataset(root='datasets/Muta/'+ config.dataset, name=config.dataset)
        test_dataset = dataset[150:]
        dataset = dataset[:150]
        kf = KFold(n_splits = 5,shuffle=True,random_state=42)
        fold = []
        for train_index, test_index in kf.split(list(range(len(dataset)))):
            fold.append((train_index, test_index))
        train_index, test_index = fold[config.fold]
        train_dataset = dataset[train_index.tolist()]
        train_dataset = train_dataset[:config.dataset_size]
        val_dataset = dataset[test_index.tolist()]
    elif config.dataset == "DrugNode":
        dataset = MutaDataset(root='datasets/Muta/DrugNode')
        datalist = dataset.data_list
        test_dataset = datalist[150:]
        dataset = dataset[:150]
        kf = KFold(n_splits = 5,shuffle=True,random_state=42)
        fold = []
        for train_index, test_index in kf.split(list(range(len(datalist)))):
            fold.append((train_index, test_index))
        train_index, test_index = fold[config.fold]

        train_dataset = [datalist[i] for i in train_index.tolist()]
        train_dataset = train_dataset[:config.dataset_size]
        val_dataset = [datalist[i] for i in test_index.tolist()]
        
        print(type(train_dataset))
        print(type(val_dataset))

    elif config.dataset in ["FullBoard","FullDiag","Simple"]:
        dataset = KRKDataset(root='datasets/KRK/'+ config.dataset)
        data_list = dataset.data_list
        state = torch.get_rng_state()
        torch.manual_seed(50)
        num_data = len(data_list)
        perm = torch.randperm(num_data).tolist()
        #print("perm: ", perm)
        data_list = [data_list[i] for i in perm]
    
        test_dataset = data_list[650:]
        data_list = data_list[:650]
        print("Test length: ", len(test_dataset))
        print("amount of positives: ", len([data for data in test_dataset if torch.equal(data.y,torch.tensor([1]))]))
        print("amount of negatives: ", len([data for data in test_dataset if torch.equal(data.y,torch.tensor([0]))]))
        
        train_dataset,val_dataset = get_fold(data_list,config.fold)
        samples = torch.randperm(len(train_dataset))[:config.dataset_size]
        train_dataset = [train_dataset[i] for i in samples.tolist()]
        print("Train length: ", len(train_dataset))
        print("amount of positives: ", len([data for data in train_dataset if torch.equal(data.y,torch.tensor([1]))]))
        print("amount of negatives: ", len([data for data in train_dataset if torch.equal(data.y,torch.tensor([0]))]))
        print("Val length: ", len(val_dataset))
        print("amount of positives: ", len([data for data in val_dataset if torch.equal(data.y,torch.tensor([1]))]))
        print("amount of negatives: ", len([data for data in val_dataset if torch.equal(data.y,torch.tensor([0]))]))
        torch.set_rng_state(state)
        
    else:
        dataset = BongardDataset(root='datasets/Bongard/'+ config.dataset)
        dataset.process()

        test_dataset = dataset[300:]
        # remove test data from dataset
        indices = np.arange(300,len(dataset),1)
        dataset.remove_datapoints(indices)
        print("Test length: ", len(test_dataset))

        real_dataset_size = math.ceil(config.dataset_size/0.8)
        indices = np.arange(real_dataset_size,len(dataset),1)
        dataset.remove_datapoints(indices)

        print("Dataset length: ", len(dataset))
        train_dataset,val_dataset = dataset.get_fold(config.fold)
        print("Train length: ", len(train_dataset))
        print("Val length: ", len(val_dataset))

    print("Type: ", type(train_dataset))
    print("Type: ", type(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    models = modelsMetadata().models
    model = models[config.model](dataset, config.layers, config.hidden_dims).to(device)
    model.reset_parameters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.start_lr)
    min_lr = 0.00001
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5,
                                                        min_lr=min_lr)

    for epoch in range(config.epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train(train_loader, model, optimizer, device)

        val_acc = test(val_loader, model, device)
        scheduler.step(val_acc)

        train_acc = test(train_loader, model, device)
        test_acc = test(test_loader, model, device)

        wandb.log({"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc,"lr":lr,"epoch":epoch})


if __name__ == "__main__":
    train_wandb()

    