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

        self.element = ['cl', 'c', 'h', 'o', 's', 'n', 'p', 'na', 'br', 'f', 'i', 'sn','pb', 'te', 'ca', 'zn', 'si', 'b', 'k', 'cu', 'y']
        self.elem_encoder = {elem: i for i, elem in enumerate(self.element)}
        self.bond_encoder = {"-": -1, "=": -2, "#": -3}


    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # current connected table should only contain the current atoms in atom_id col and atom_id2 col
            current_kb["connected"] = current_kb["connected"][current_kb["connected"]["atom_id"].isin(current_kb["atom"]["atom_id"]) & current_kb["connected"]["atom_id2"].isin(current_kb["atom"]["atom_id"])]
            current_kb["bond"] = current_kb["bond"][current_kb["bond"]["bond_id"].isin(current_kb["connected"]["bond_id"])]

            # create node features
            num_of_node_features = len(self.elem_encoder) + 3 # 3 bond types
            num_nodes = len(current_kb["atom"]) + len(current_kb["bond"])

            node_features = torch.zeros((num_nodes, num_of_node_features))
            
            index = 0
            for _,row in current_kb["atom"].iterrows():
                node_features[index][self.elem_encoder[row["element"]]] = 1
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

        # atom, bond, connected
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # current connected table should only contain the current atoms in atom_id col and atom_id2 col
            current_kb["connected"] = current_kb["connected"][current_kb["connected"]["atom_id"].isin(current_kb["atom"]["atom_id"]) & current_kb["connected"]["atom_id2"].isin(current_kb["atom"]["atom_id"])]

            # create node features
            num_of_node_features = len(self.elem_encoder)
            num_nodes = len(current_kb["atom"])
            
            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            index = 0
            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][self.elem_encoder[row["element"]]] = 1
                index += 1
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            # assumption that all the benzene and nitro atom_ids are already in the bond table
            for _,row in current_kb["connected"].iterrows():
                node_id_1 = node_mapping[row["atom_id"]]
                node_id_2 = node_mapping[row["atom_id2"]]
                edge_feat = [0,0,0]
                bond_type = current_kb["bond"][current_kb["bond"]["bond_id"] == row["bond_id"]]["bond_type"].values[0]
                if bond_type == "-":
                    edge_feat[0] = 1
                elif bond_type == "=":
                    edge_feat[1] = 1
                elif bond_type == "#":
                    edge_feat[2] = 1
                self.add_edges(edge_indices,node_id_1,node_id_2,edge_features,edge_feat)
                
            
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
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

            # current connected table should only contain the current atoms in atom_id col and atom_id2 col
            current_kb["connected"] = current_kb["connected"][current_kb["connected"]["atom_id"].isin(current_kb["atom"]["atom_id"]) & current_kb["connected"]["atom_id2"].isin(current_kb["atom"]["atom_id"])]

            num_of_node_features = len(self.elem_encoder) + 1 # 1 for instance feature
            num_nodes = len(current_kb["atom"]) + len(current_kb["atom"]["element"].unique())

            node_features = torch.zeros((num_nodes, num_of_node_features))

            node_index = 0
            atom_mapping = {}
            for atom_type in current_kb["atom"]["element"].unique():
                node_features[node_index][self.elem_encoder[atom_type]] = 1
                atom_mapping[atom_type] = node_index
                node_index += 1
            
            # create the edges
            edge_indices = [[],[]]
            edge_features = []
            for _,row in current_kb["atom"].iterrows():
                node_features[node_index][-1] = 1 # instance node
                atom_mapping[row["atom_id"]] = node_index
                atom_type_index = atom_mapping[row["element"]]
                self.add_edges(edge_indices,atom_type_index,node_index)
                edge_features.append([1,0,0,0]) # external edge
                edge_features.append([1,0,0,0]) # external edge
                node_index += 1
            
            for _,row in current_kb["connected"].iterrows():
                node_index1 = atom_mapping[row["atom_id"]]
                node_index2 = atom_mapping[row["atom_id2"]]
                edge_feat = [0,0,0,0]
                self.add_edges(edge_indices,node_index1,node_index2)
                bond_type = self.bond_encoder[current_kb["bond"][current_kb["bond"]["bond_id"] == row["bond_id"]]["bond_type"].values[0]] 
                edge_feat[bond_type] = 1
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

            # current connected table should only contain the current atoms in atom_id col and atom_id2 col
            current_kb["connected"] = current_kb["connected"][current_kb["connected"]["atom_id"].isin(current_kb["atom"]["atom_id"]) & current_kb["connected"]["atom_id2"].isin(current_kb["atom"]["atom_id"])]

            num_node_features = len(self.elem_encoder) + len(self.bond_encoder)
            num_nodes = len(current_kb["atom"]) + len(current_kb["connected"])
            
            node_features = torch.zeros((num_nodes, num_node_features))

            edge_indices = [[],[]]

            node_mapping = {}
            index = 0

            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][self.elem_encoder[row["element"]]] = 1
                index += 1
            
            for _,row in current_kb["connected"].iterrows():
                bond_type = self.bond_encoder[current_kb["bond"][current_kb["bond"]["bond_id"] == row["bond_id"]]["bond_type"].values[0]]
                node_features[index][bond_type] = 1

                node_id_1 = node_mapping[row["atom_id"]]
                node_id_2 = node_mapping[row["atom_id2"]]
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