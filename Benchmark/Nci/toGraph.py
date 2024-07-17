import pandas as pd
import os
import glob
import torch_geometric
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.data import HeteroData
import re

class toGraph:
    def __init__(self, relational_path, dataset_name, dataset_problem_key,target):
        self.relational_path = relational_path
        self.dataset_name = dataset_name
        self.problem_key = dataset_problem_key
        self.target = target
        self.kb = self.collect_relational()
        self.num_of_graphs = len(self.kb[self.dataset_name][self.problem_key].unique())
        self.y = self.create_y(self.kb)
        self.symbol_to_number = {
                        "O": 0,
                        "N": 1,
                        "C": 2,
                        "S": 3,
                        "Cl": 4,
                        "P": 5,
                        "F": 6,
                        "Na": 7,
                        "Sn": 8,
                        "Pt": 9,
                        "Ni": 10,
                        "Zn": 11,
                        "Mn": 12,
                        "Br": 13,
                        "Cu": 14,
                        "Co": 15,
                        "Se": 16,
                        "Au": 17,
                        "Pb": 18,
                        "Ge": 19,
                        "I": 20,
                        "Si": 21,
                        "Fe": 22,
                        "Cr": 23,
                        "Hg": 24,
                        "As": 25,
                        "B": 26,
                        "Ga": 27,
                        "Ti": 28,
                        "Bi": 29,
                        "Y": 30,
                        "Nd": 31,
                        "Eu": 32,
                        "Tl": 33,
                        "Zr": 34,
                        "Hf": 35,
                        "In": 36,
                        "K": 37,
                        "La": 38,
                        "Ce": 39,
                        "Sm": 40,
                        "Gd": 41,
                        "Dy": 42,
                        "U": 43,
                        "Pd": 44,
                        "Ir": 45,
                        "Re": 46,
                        "Li": 47,
                        "Sb": 48,
                        "W": 49,
                        "Mg": 50,
                        "Ru": 51,
                        "Rh": 52,
                        "Os": 53,
                        "Th": 54,
                        "Mo": 55,
                        "Nb": 56,
                        "Ta": 57,
                        "Ag": 58,
                        "Cd": 59,
                        "Er": 60,
                        "V": 61,
                        "Ac": 62,
                        "Te": 63,
                        "Al": 64,
                    }

    def collect_relational(self):
        kb = {}
        predicates = []
        all_files = glob.glob(os.path.join(self.relational_path, "*.csv"))
        for filename in all_files:
            df = pd.read_csv(filename)
            relation = filename.split('/')[-1].split('.')[0]
            predicates.append(relation)
            kb[relation] = df
        # sort all the dataframes by the problemId
        for key in kb.keys():
            kb[key] = kb[key].sort_values(by=[self.problem_key])
        return kb
    
    def create_y(self, kb):
        y = kb[self.dataset_name][self.target].to_numpy()
        y[y == "neg"] = 0
        y[y == " neg"] = 0
        y[y == "pos"] = 1
        y[y == " pos"] = 1
        return y
    
    def add_edges(self,edge_indices,node_index1,node_index2):
        edge_indices[0].append(node_index1)
        edge_indices[1].append(node_index2)
        edge_indices[0].append(node_index2)
        edge_indices[1].append(node_index1)
    
    
    def get_current_objects(self,index):
        # also removes the problem key column from the dataframe
        current_kb = {}
        for key in self.kb.keys():
            if key != self.dataset_name:
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
            if key == self.dataset_name:
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
        return current_kb
    
    # CHECKED 
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)


            # assumption that every table has an problemId column
            num_nodes = len(current_kb["atom"]) + len(current_kb["bond"])


            # create node feature array
            num_node_features = len(self.symbol_to_number) + 3 # 3 for the possible edges
            node_features = np.zeros((num_nodes, num_node_features))

            # create edge indices array : fully bidirectionally connected graph
            amount_of_edges = (num_nodes * (num_nodes - 1))
            edge_index = np.zeros((2,amount_of_edges))

            # create edge feature array
            edge_features = np.ones((amount_of_edges,1))

            # create node features
            total_index = 0
            for _,row in current_kb["atom"].iterrows():
                node_features[total_index][self.symbol_to_number[row["atom_type"]]] = 1
                total_index += 1
            for _,row in current_kb["bond"].iterrows():
                node_features[total_index][-row["bond_type"]] = 1
                total_index += 1
            
            # fill in the edge indices
            index = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index[0][index] = i
                        edge_index[1][index] = j
                        index += 1
            
            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = torch.tensor(edge_index, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
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
            num_of_node_features = len(self.symbol_to_number)
            num_nodes = len(current_kb["atom"])
            
            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            index = 0
            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][self.symbol_to_number[row["atom_type"]]] = 1
                index += 1
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            # assumption that all the benzene and nitro atom_ids are already in the bond table
            for key in edge_predicates:
                if key == "bond":
                    for _,row in current_kb[key].iterrows():
                        node_index1 = node_mapping[row["atom1"]]
                        node_index2 = node_mapping[row["atom2"]]
                        edge_feat = [0,0,0]
                        edge_feat[row["bond_type"]-1] = 1
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
        problem_key = "problemId"
        data_list = []
        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # 3 for the external predicates and 1 for instances
            num_node_features = len(self.symbol_to_number) + 1  # added the instance feature

            # external and internal predicates
            num_edge_features = 4 # 1 for external and 3 for bond types
            
            num_nodes = len(current_kb["atom"]) + len(current_kb["atom"]["atom_type"].unique())
            # create the node features 
            node_features = np.zeros((num_nodes, num_node_features))

            node_i = 0
            atom_mapping = {}
            for atom_t in current_kb["atom"]["atom_type"].unique():
                node_features[node_i][self.symbol_to_number[atom_t]] = 1
                atom_mapping[atom_t] = node_i
                node_i += 1
            

            # create the edges
            edge_indices = [[],[]]
            edge_features = []

            # Fill in external edges and the node feature
            
            for _,row in current_kb["atom"].iterrows():
                node_features[node_i][-1] = 1
                atom_mapping[row["atom_id"]] = node_i
                atom_type_index = atom_mapping[row["atom_type"]]
                self.add_edges(edge_indices,atom_type_index,node_i)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
                node_i += 1
                
            
            # fill in the internal edges
            for index,row in current_kb["bond"].iterrows():
                node_index1 = atom_mapping[row["atom1"]]
                node_index2 = atom_mapping[row["atom2"]]
                edge_feat = [0,0,0,0]
                self.add_edges(edge_indices,node_index1,node_index2)
                edge_feat[row["bond_type"]] = 1
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label))

        return data_list
    
    def Klog(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            num_node_features = len(self.symbol_to_number) + 3 # 3 for the possible edges
            num_nodes = len(current_kb["atom"]) + len(current_kb["bond"])
            node_features = np.zeros((num_nodes, num_node_features))

            edge_indices = [[],[]]
            
            # create the node features from "atom"
            node_mapping = {}
            index = 0

            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][self.symbol_to_number[row["atom_type"]]] = 1
                index += 1
            
            for _,row in current_kb["bond"].iterrows():
                bond_type = row["bond_type"]
                node_features[index][-bond_type] = 1
                node_id1 = node_mapping[row["atom1"]] 
                node_id2 = node_mapping[row["atom2"]]
                self.add_edges(edge_indices,node_id1,index)
                self.add_edges(edge_indices,node_id2,index)
                index += 1
            
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(np.ones((len(edge_indices[0]),1)), dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)
        
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=truth_label))

        return data_list
            
    