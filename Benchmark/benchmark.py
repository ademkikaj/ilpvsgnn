import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from gnn_baselines.gnn_architectures import modelsMetadata
import random
import torch
import time
import numpy as np

class Benchmark:

    def __init__(self) -> None:
        pass

    def prepare_csv(self,relational_path):
        # load all csv file in relational directory
        keys = []
        for file in os.listdir(relational_path):
            if file.endswith(".csv"):
                keys.append(file[:-4])
        
        for key in keys:
            with open(relational_path + f"/{key}.csv",'r') as f:
                lines = f.readlines()
            with open(relational_path + f"/{key}.csv",'w') as f:
                for line in lines:
                    f.write(line.replace(';',','))
        return
    
    def split_relational_data(self,relational_path,dataset_name,class_name,problem_id):
        # load the main data
        df = pd.read_csv(relational_path + f"/{dataset_name}.csv")
        y = df[class_name]
        train, test ,y_train,y_test = train_test_split(df,y,stratify=y,test_size=0.21,random_state=41)
        train_problem_ids = train[problem_id]
        
        kb = {}
        # load all csv file in relational directory
        for file in os.listdir(relational_path):
            if file.endswith(".csv"):
                kb[file[:-4]] = pd.read_csv(relational_path + f"/{file}")
        train = {}
        test = {}
        for key in kb.keys():
            if problem_id not in kb[key].columns:
                train[key] = kb[key]
                test[key] = kb[key]
            else:
                df = kb[key]
                train[key] = df[df[problem_id].isin(train_problem_ids)]
                test[key] = df[~df[problem_id].isin(train_problem_ids)]
        
        # write them away
        for key in train.keys():
            
            train[key].to_csv(relational_path + "/train" + f"/{key}.csv",index=False)
            test[key].to_csv(relational_path + "/test"  + f"/{key}.csv",index=False)
    
    def new_split_relational_data(self,relational_path,dataset_name,class_name,problem_id):
        df = pd.read_csv(relational_path + f"/{dataset_name}.csv")
        y = df[class_name]
        train, test ,y_train,y_test = train_test_split(df,y,stratify=y,test_size=0.2)
        train_problem_ids = train[problem_id]
        
        kb = {}
        # load all csv file in relational directory
        for file in os.listdir(relational_path):
            if file.endswith(".csv"):
                kb[file[:-4]] = pd.read_csv(relational_path + f"/{file}")

        train = {}
        test = {}
        df = kb[dataset_name]
        for key in kb.keys():
            if key != dataset_name:
                train[key] = kb[key]
                test[key] = kb[key]
            else:
                df = kb[key]
                train[key] = df[df[problem_id].isin(train_problem_ids)]
                test[key] = df[~df[problem_id].isin(train_problem_ids)]
        
        # write them away
        for key in train.keys():
            train[key].to_csv(relational_path + "/train" + f"/{key}.csv",index=False)
            test[key].to_csv(relational_path + "/test"  + f"/{key}.csv",index=False)

    


    def run_gnn(self,graphs,test_graphs,model,layers,hidden_dims,config) -> pd.DataFrame:
        # no use of the folds, just multiple repetitions

        # variables
        model_name = model
        layers = layers
        hidden_dims = hidden_dims
        
        epochs = 200
        
        min_lr = 0.00001
        batch_size = 16

        # folds = self.config['folds']
        repititions = config['repetitions']

        num_node_features = graphs[0].num_node_features
        num_classes = 2
        num_edge_features = graphs[0].num_edge_features
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        test_accuracies = []
        runtimes = []

        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=True)

        for i in range(repititions):
            print("Repetition: ", i+1)

            random.shuffle(graphs)
            train_index, val_index = train_test_split(range(len(graphs)),test_size=0.1)
                
            best_val_acc = 0.0
            best_test_acc = 0.0

            train_dataset = [graphs[i] for i in train_index]
            val_dataset = [graphs[i] for i in val_index]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

            models = modelsMetadata().models
            model = models[model_name](num_node_features,num_edge_features,num_classes,layers, hidden_dims).to(device)
            model.reset_parameters()

            lr = 0.01
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5,min_lr=min_lr)

            train_data = pd.DataFrame()
            # Time the training of the GNN
            start = time.time()
            for epoch in tqdm(range(epochs),desc="Epochs"):
                lr = scheduler.optimizer.param_groups[0]['lr']
                train_loss = self.train(train_loader, model, optimizer, device)
                val_acc = self.test(val_loader, model, device)
                scheduler.step(val_acc)
                train_acc = self.test(train_loader, model, device)
                test_acc = self.test(test_loader, model, device)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                
                if lr <= min_lr:
                    break

            end = time.time()

            runtimes.append(end-start)

            test_accuracies.append(best_test_acc)
        
        # test accuracies over all the repetitions
        test_acc = np.array(test_accuracies).mean()
        test_acc_std = np.array(test_accuracies).std()

        runtime = np.array(runtimes).mean()

        train_data = pd.DataFrame({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'test_acc_std': test_acc_std,
            'lr': lr,
            'epoch': epoch,
            'model': model,
            'layers': layers,
            'hidden_dims': hidden_dims,
            'total_epochs': epochs,
            'runtime': runtime,
            'test_size': len(test_loader),
            'val_size': len(val_dataset),
            'train_size': len(train_dataset)
            },index=[0])
        
        return train_data
    
     # One training epoch for GNN model.
    def train(self,train_loader, model, optimizer, device):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        return total_loss  

    # Get acc. of GNN model.
    def test(self,loader, model, device):
        model.eval()

        correct = 0
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)