import pandas as pd
import os
import glob
import torch_geometric
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.data import HeteroData
import re
from ..GraphConversion import GraphConversion

class toGraph(GraphConversion):

    def __init__(self, relational_path, dataset_name, dataset_problem_key,target):
        super().__init__(relational_path, dataset_name, dataset_problem_key,target)
        self.atom_encoder = {120: 0, 113: 1, 121: 2, 83: 3, 96: 4, 3: 5, 1: 6, 8: 7, 2: 8, 84: 9, 94: 10, 95: 11, 134: 12, 115: 13, 85: 14, 32: 15, 36: 16, 35: 17, 38: 18, 31: 19, 34: 20, 33: 21, 37: 22, 499: 23, 40: 24, 45: 25, 51: 26, 52: 27, 50: 28, 49: 29, 42: 30, 41: 31, 53: 32, 81: 33, 22: 34, 14: 35, 16: 36, 10: 37, 21: 38, 191: 39, 17: 40, 19: 41, 26: 42, 193: 43, 192: 44, 27: 45, 29: 46, 232: 47, 15: 48, 77: 49, 72: 50, 70: 51, 76: 52, 74: 53, 75: 54, 78: 55, 79: 56, 93: 57, 101: 58, 87: 59, 129: 60, 62: 61, 61: 62, 60: 63, 92: 64, 102: 65}
        self.bond_encoder = {1: -1, 2: -2, 3: -3, 7: -4}

    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = len(self.atom_encoder) + len(self.bond_encoder) # 4 bond types
            num_nodes = len(current_kb["atom"]) + len(current_kb["bond"])

            node_features = torch.zeros((num_nodes, num_of_node_features))
            
            index = 0
            for _,row in current_kb["atom"].iterrows():
                node_features[index][self.atom_encoder[row["atom_type"]]] = 1
                index += 1
            
            for _,row in current_kb["bond"].iterrows():
                node_features[index][self.bond_encoder[row["bond_type"]]] = 1
                index += 1
            
            # fully connect the nodes
            edge_indices = [[],[]]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        self.add_edges(edge_indices,i,j)
            
            data_list.append(Data(
                x = node_features,
                edge_index = torch.tensor(edge_indices, dtype=torch.int64),
                edge_attr = torch.ones(len(edge_indices[0]),1,dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list

    def node_edge(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features

        node_predicates = ["atom"]
        edge_predicates = ["bond"]
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = len(self.atom_encoder)
            num_nodes = len(current_kb["atom"])
            
            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            index = 0
            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][self.atom_encoder[row["atom_type"]]] = 1
                index += 1
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            # assumption that all the benzene and nitro atom_ids are already in the bond table
            for key in edge_predicates:
                if key == "bond":
                    for _,row in current_kb[key].iterrows():
                        node_index1 = node_mapping[row["atom_id_1"]]
                        node_index2 = node_mapping[row["atom_id_2"]]
                        edge_feat = [0,0,0,0]
                        if row["bond_type"] == 1:
                            edge_feat[0] = 1
                        elif row["bond_type"] == 2:
                            edge_feat[1] = 1
                        elif row["bond_type"] == 3:
                            edge_feat[2] = 1
                        elif row["bond_type"] == 7:
                            edge_feat[3] = 1
                        self.add_edges(edge_indices,node_index1,node_index2)
                        edge_features.append(edge_feat)
                        edge_features.append(edge_feat)
            
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)

            # create edge features
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = edge_features,
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list

    def edge_based(self):
        data_list = []
        
        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            num_of_node_features = len(self.atom_encoder) + 1 # 1 for instance feature
            num_nodes = len(current_kb["atom"]) + len(current_kb["atom"]["atom_type"].unique())

            node_features = torch.zeros((num_nodes, num_of_node_features))

            node_index = 0
            atom_mapping = {}
            for atom_type in current_kb["atom"]["atom_type"].unique():
                node_features[node_index][self.atom_encoder[atom_type]] = 1
                atom_mapping[atom_type] = node_index
                node_index += 1
            
            # create the edges
            edge_indices = [[],[]]
            edge_features = []
            for _,row in current_kb["atom"].iterrows():
                node_features[node_index][-1] = 1 # instance node
                atom_mapping[row["atom_id"]] = node_index
                atom_type_index = atom_mapping[row["atom_type"]]
                self.add_edges(edge_indices,atom_type_index,node_index)
                edge_features.append([1,0,0,0,0])
                edge_features.append([1,0,0,0,0])
                node_index += 1
            
            for _,row in current_kb["bond"].iterrows():
                node_index1 = atom_mapping[row["atom_id_1"]]
                node_index2 = atom_mapping[row["atom_id_2"]]
                edge_feat = [0,0,0,0,0]
                self.add_edges(edge_indices,node_index1,node_index2)
                edge_feat[self.bond_encoder[row["bond_type"]]*-1] = 1
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                node_index += 1
            
            data_list.append(Data(
                x = node_features,
                edge_index = torch.tensor(edge_indices, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))

        return data_list

    def Klog(self):
        data_list = []

        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)
    
            num_node_features = len(self.atom_encoder) + len(self.bond_encoder)
            num_nodes = len(current_kb["atom"]) + len(current_kb["bond"])
            
            node_features = torch.zeros((num_nodes, num_node_features))

            edge_indices = [[],[]]

            node_mapping = {}
            index = 0

            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][self.atom_encoder[row["atom_type"]]] = 1
                index += 1
            
            for _,row in current_kb["bond"].iterrows():
                bond_type = self.bond_encoder[row["bond_type"]]
                node_features[index][bond_type] = 1
                node_id_1 = node_mapping[row["atom_id_1"]]
                node_id_2 = node_mapping[row["atom_id_2"]]
                self.add_edges(edge_indices,node_id_1,index)
                self.add_edges(edge_indices,node_id_2,index)
                index += 1
            
            data_list.append(Data(
                x = node_features,
                edge_index = torch.tensor(edge_indices, dtype=torch.int64),
                edge_attr = torch.ones(len(edge_indices[0]),1,dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list