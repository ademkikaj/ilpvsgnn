from typing import Tuple, Union
from torch import Tensor
from torch_geometric.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data import Data   
from sklearn.model_selection import KFold
import os.path as osp
import os
import torch


class KRKDataset(Dataset):
    def __init__(self, root, datalist=None,transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        if datalist is not None:
            self.data_list = datalist
            self.skip_process = True
        else:
            self.data_list = torch.load(self.raw_paths[0])
            self.skip_process = False
    
    @property
    def raw_file_names(self):
        return ['datalist.pt']
    
    @property
    def processed_file_names(self):
        return ['data_'+str(i)+'.pt' for i in range(0, 999)]
    
    @property
    def num_features(self) -> int:
        return self.data_list[0].x.shape[1]
    
    @property
    def num_classes(self) -> int:
        return len(set([data.y.item() for data in self.data_list]))
    
    def process(self):
        if not self.skip_process:
            print("Processing dataset")
            for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                self.data_list = torch.load(raw_path)
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                
                for idx, data in enumerate(self.data_list):
                    torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
    
    def len(self) -> int:
        return len(self.data_list)

    def get_y(self):
        return [data.y.item() for data in self.data_list]
    
    def get_x(self):
        return torch.zeros((len(self.data_list),self.num_features))
    
    def get(self, idx):
        data = self.data_list[idx]
        return data

    def get_amount_of_positives(self):
        return len([data for data in self.data_list if data.y == 1])
    
    def get_amount_of_negatives(self):
        return len([data for data in self.data_list if data.y == 0])

    def remove_datapoints(self,indices):
        # remove datapoints from the datasetlist
        self.data_list = [data for i, data in enumerate(self.data_list) if i not in indices]

    def getDatasetOverview(self):
        amount_of_positives = len([data for data in self.data_list if data.y == 1])
        amount_of_negatives = len([data for data in self.data_list if data.y == 0])
        amount_of_examples = len(self.data_list)
        average_amount_of_nodes = sum([data.num_nodes for data in self.data_list]) / amount_of_examples
        average_amount_of_edges = sum([data.num_edges for data in self.data_list]) / amount_of_examples
        overview = {"amount_of_examples": amount_of_examples, "amount_of_positives": amount_of_positives,
                    "amount_of_negatives": amount_of_negatives, "average_amount_of_nodes": average_amount_of_nodes,
                    "average_amount_of_edges": average_amount_of_edges}
        return overview
    
    ### Helper functions for different representations of the datast

    def is_white_king(self,node):
        possibilities = torch.tensor([[1,0,0],[1,1,0],[1,0,1],[1,1,1]],dtype=torch.int32)
        for row in possibilities:
            if torch.equal(node,row):
                return True
        return False

    def is_black_king(self,node):
        possibilities = torch.tensor([[0,0,1],[0,1,1],[1,0,1],[1,1,1]],dtype=torch.int32)
        for row in possibilities:
            if torch.equal(node,row):
                return True
        return False
    

    def create_folds(self,k):
        self.folds = []
        kf = KFold(n_splits = k,shuffle=True,random_state=42)
        for train_index, test_index in kf.split(self.data_list):
            self.folds.append((train_index, test_index))
        
    def get_fold(self, fold):
        self.create_folds(5)
        train_index, test_index = self.folds[fold]
        train = self[train_index.tolist()]
        test = self[test_index.tolist()]
        return train, test

        

