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
    def __init__(self, relational_path, dataset_name, dataset_problem_key,target,name_encoder):
        self.relational_path = relational_path
        self.dataset_name = dataset_name
        self.problem_key = dataset_problem_key
        self.target = target
        self.kb = self.collect_relational()
        self.num_of_graphs = len(self.kb[self.dataset_name])
        self.num_node_features = 3
        self.y = self.create_y(self.kb)
        self.name_encoder = name_encoder 

    # changed for this current University dataset
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
        # no sorting needed
        return kb
    
    def create_y(self, kb):
        y = kb[self.dataset_name][self.target].to_numpy()
        y[y == "pos"] = 1
        y[y == "neg"] = 0
        return y
    
    def add_edges(self,edge_indices,node_index1,node_index2):
        edge_indices[0].append(node_index1)
        edge_indices[1].append(node_index2)
        edge_indices[0].append(node_index2)
        edge_indices[1].append(node_index1)
    
    def get_current_objects(self,name,index):
        # also removes the problem key column from the dataframe
        current_kb = {}
        current_kb[self.dataset_name] = self.kb[self.dataset_name].iloc[index]
        name1 = current_kb[self.dataset_name]["name1"]
        name2 = current_kb[self.dataset_name]["name2"]

        for key in self.kb.keys():
            if key == "same_gen":
                # all rows where either name1 or name2 is in the row
                current_kb[key] = self.kb[key][(self.kb[key]["name1"].isin([name1, name2])) | (self.kb[key]["name2"].isin([name1, name2]))]

            elif key == "parent":
                current_kb[key] = self.kb[key][(self.kb[key]["name1"].isin([name1, name2])) | (self.kb[key]["name2"].isin([name1, name2]))]

        return current_kb
    
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id,graph_index)
            
            # create node features
            num_node_features = 2 + len(self.name_encoder)

            all_names = []
            all_names.append(current_kb[self.dataset_name]["name1"])
            all_names.append(current_kb[self.dataset_name]["name2"])
            all_names += list(current_kb["parent"]["name1"].unique())
            all_names += list(current_kb["parent"]["name2"].unique())
            all_names +=list(current_kb["same_gen"]["name1"].unique())
            all_names += list(current_kb["same_gen"]["name2"].unique())
            num_nodes = len(all_names)

            node_features = np.zeros((num_nodes, num_node_features))

            node_mapping = {}

            index = 0
            # add the student features
             # add the student features
            for name in all_names:
                node_features[index][:] = self.name_encoder[name]
                node_mapping[name] = index
                index += 1
            
           
            edge_indices = [[],[]]
            # fully connect the edges
            for i in range(num_nodes):
                for j in range(i,num_nodes):
                    if i != j:
                        self.add_edges(edge_indices,i,j)
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.ones((edge_index.shape[1],1), dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list

    def node_edge(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id,graph_index)
            
            # create node features
            num_node_features = len(self.name_encoder)
            # get all unique names
            all_names = []
            all_names.append(current_kb[self.dataset_name]["name1"])
            all_names.append(current_kb[self.dataset_name]["name2"])
            all_names += list(current_kb["parent"]["name1"].unique())
            all_names += list(current_kb["parent"]["name2"].unique())
            all_names +=list(current_kb["same_gen"]["name1"].unique())
            all_names += list(current_kb["same_gen"]["name2"].unique())

            num_nodes = len(all_names)

            node_features = np.zeros((num_nodes, num_node_features))

            node_mapping = {}
            index = 0
            # add the student features
            for name in all_names:
                node_features[index][:] = self.name_encoder[name]
                node_mapping[name] = index
                index += 1
           
            edge_indices = [[],[]]
            edge_features = []
            # add the edges
            for _,row in current_kb["parent"].iterrows():
                name1 = row["name1"]
                name2 = row["name2"]
                name1_index = node_mapping[name1]
                name2_index = node_mapping[name2]
                self.add_edges(edge_indices,name1_index,name2_index)
                edge_features.append([1,0])
                edge_features.append([1,0])
            
            for _,row in current_kb["same_gen"].iterrows():
                name1 = row["name1"]
                name2 = row["name2"]
                name1_index = node_mapping[name1]
                name2_index = node_mapping[name2]
                self.add_edges(edge_indices,name1_index,name2_index)
                edge_features.append([0,1])
                edge_features.append([0,1])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list
    
    def edge_based(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id,graph_index)
            
            # create node features
            num_node_features = 1 + len(self.name_encoder)
            # get all unique names
            all_names = []
            all_names.append(current_kb[self.dataset_name]["name1"])
            all_names.append(current_kb[self.dataset_name]["name2"])
            all_names += list(current_kb["parent"]["name1"].unique())
            all_names += list(current_kb["parent"]["name2"].unique())
            all_names +=list(current_kb["same_gen"]["name1"].unique())
            all_names += list(current_kb["same_gen"]["name2"].unique())

            num_nodes = len(all_names)*2 

            node_features = np.zeros((num_nodes, num_node_features))

            node_mapping = {}
            index = 0
            edge_indices = [[],[]]
            edge_features = []
            # add the student features
            for name in all_names:
                node_features[index][1:] = self.name_encoder[name]
                index += 1
                node_features[index][0] = 1
                node_mapping[name] = index
                index += 1
                self.add_edges(edge_indices,index-2,index-1)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
           
            
            # add the edges
            for _,row in current_kb["parent"].iterrows():
                name1 = row["name1"]
                name2 = row["name2"]
                name1_index = node_mapping[name1]
                name2_index = node_mapping[name2]
                self.add_edges(edge_indices,name1_index,name2_index)
                edge_features.append([0,1,1,0])
                edge_features.append([0,1,1,0])
            
            for _,row in current_kb["same_gen"].iterrows():
                name1 = row["name1"]
                name2 = row["name2"]
                name1_index = node_mapping[name1]
                name2_index = node_mapping[name2]
                self.add_edges(edge_indices,name1_index,name2_index)
                edge_features.append([0,1,0,1])
                edge_features.append([0,1,0,1])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list

    def Klog(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id,graph_index)
            
            # create node features
            num_node_features = 2 + len(self.name_encoder)
            # get all unique names
            all_names = []
            all_names.append(current_kb[self.dataset_name]["name1"])
            all_names.append(current_kb[self.dataset_name]["name2"])
            all_names += list(current_kb["parent"]["name1"].unique())
            all_names += list(current_kb["parent"]["name2"].unique())
            all_names +=list(current_kb["same_gen"]["name1"].unique())
            all_names += list(current_kb["same_gen"]["name2"].unique())

            num_nodes = len(all_names) + len(current_kb["parent"]) + len(current_kb["same_gen"])

            node_features = np.zeros((num_nodes, num_node_features))

            node_mapping = {}
            index = 0
            # add the student features
            for name in all_names:
                node_features[index][2:] = self.name_encoder[name]
                node_mapping[name] = index
                index += 1
           
            edge_indices = [[],[]]
            edge_features = []
            # add the edges
            for _,row in current_kb["parent"].iterrows():
                name1 = row["name1"]
                name2 = row["name2"]
                name1_index = node_mapping[name1]
                name2_index = node_mapping[name2]
                node_features[index][0] = 1
                self.add_edges(edge_indices,name1_index,index)
                edge_features.append([1,0])
                edge_features.append([1,0])
                self.add_edges(edge_indices,index,name2_index)
                edge_features.append([1,0])
                edge_features.append([1,0])
                index += 1
            
            for _,row in current_kb["same_gen"].iterrows():
                name1 = row["name1"]
                name2 = row["name2"]
                name1_index = node_mapping[name1]
                name2_index = node_mapping[name2]
                node_features[index][1] = 1
                self.add_edges(edge_indices,name1_index,index)
                edge_features.append([0,1])
                edge_features.append([0,1])
                self.add_edges(edge_indices,index,name2_index)
                edge_features.append([0,1])
                edge_features.append([0,1])
                index += 1
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list

    
