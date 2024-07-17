from typing import Callable, Optional
from torch_geometric.data import Dataset
import torch
from torch_geometric.data import Batch
from sklearn.model_selection import KFold
import os.path as osp
import os

import torch
from torch_geometric.data import Dataset

class MutaDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

        self.data_list = torch.load(self.raw_paths[0])

        self.folds = None

        try:
            self.data_list[0].metadata()
            self.hetero = True
        except:
            self.hetero = False
    
    @property
    def raw_file_names(self):
        return ['datalist.pt']

    @property
    def processed_file_names(self):
        return ['data_'+str(i)+'.pt' for i in range(0, 188)]
    
    @property
    def num_features(self) -> int:
        return self.data_list[0].num_node_features

    @property
    def num_classes(self) -> int:
        return len(set([data.y.item() for data in self.data_list]))
    
    @property
    def num_edge_features(self) -> int:
        return self.data_list[0].num_edge_features

    def process(self):
        print("process")
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            self.data_list = torch.load(raw_path)
            self.test_data = self.data_list[300:350]
            
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            for idx, data in enumerate(self.data_list):
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        data = self.data_list[idx]
        return data
    
    