from torch_geometric.data import Dataset
import torch
from torch_geometric.data import Batch
from sklearn.model_selection import KFold
import os.path as osp
import os

import torch
from torch_geometric.data import Dataset

class BongardDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,datalist=None):
        super().__init__(root, transform, pre_transform)
        if datalist is not None:
            self.data_list = datalist
        else:
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
        return ['data_'+str(i)+'.pt' for i in range(0, 392)]

    @property
    def num_features(self) -> int:
        if self.hetero:
            return sum([i for i in self.data_list[0].num_node_features.values()])
        else:
            return self.data_list[0].num_node_features

    @property
    def num_classes(self) -> int:
        return len(set([data.y.item() for data in self.data_list]))
    
    @property
    def num_edge_features(self) -> int:
        if self.hetero:
            return sum([i for i in self.data_list[0].num_edge_features.values()])
        else:
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

    def get_amount_of_positives(self):
        return len([data for data in self.data_list if data.y == 1])
    
    def get_amount_of_negatives(self):
        return len([data for data in self.data_list if data.y == 0])
    
    def remove_datapoints(self, indices):
        """
        Remove datapoints with the given indices from the dataset.
        """
        if len(indices) > 0:
            # Remove from processed data
            # for idx in indices:
            #     file_path = osp.join(self.processed_dir, f'data_{idx}.pt')
            #     if osp.exists(file_path):
            #         os.remove(file_path)
            
            # Remove from data_list
            self.data_list = [data for i, data in enumerate(self.data_list) if i not in indices]

    def getDatasetOverview(self):
        amount_of_positives = len([data for data in self.data_list if data.y == 1])
        amount_of_negatives = len([data for data in self.data_list if data.y == 0])
        amount_of_examples = len(self)
        average_amount_of_nodes = sum([data.num_nodes for data in self.data_list]) / amount_of_examples
        average_amount_of_edges = sum([data.num_edges for data in self.data_list]) / amount_of_examples
        overview = {"amount_of_examples": amount_of_examples, "amount_of_positives": amount_of_positives,
                    "amount_of_negatives": amount_of_negatives, "average_amount_of_nodes": average_amount_of_nodes,
                    "average_amount_of_edges": average_amount_of_edges}
        return overview


    ### Needs testing
    def to_node_value_encoding(self):
        """
        Convert the dataset from one-hot-encoding to node value encoding.
        The structure change of the dataset must be performed first.
        """
        max_value = self.data_list[0].x.shape[1]
        value_ranges = range(1, max_value + 1)
        new_data_list = []
        for graph in self.data_list:
            new_x = torch.zeros((graph.x.shape[0], 1))
            indices = graph.x.argmax(dim=1)
            for i, index in enumerate(indices):
                new_x[i] = value_ranges[index]
            new_graph = Batch(x=new_x, edge_index=graph.edge_index, y=graph.y)
            new_data_list.append(new_graph)
        self.data_list = new_data_list

    def create_folds(self,k):
        """
        Create k folds of the dataset.
        """
        self.folds = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(list(range(len(self.data_list)))):
            self.folds.append((train_index, test_index))
    
    def get_fold(self, fold):
        self.create_folds(5)

        train_index, test_index = self.folds[fold]
        train = self[train_index.tolist()]
        test = self[test_index.tolist()]
        return train, test

    


    
    

    

            

    
