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
        self.num_node_features = 3
        self.y = self.create_y(self.kb)

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
        # normalize the drug columns lumo and logp, maximum absolut scaling
        kb["mutag"]["logp"] = kb["mutag"]["logp"]/kb["mutag"]["logp"].abs().max()
        kb["mutag"]["lumo"] = kb["mutag"]["lumo"]/kb["mutag"]["lumo"].abs().max()
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
    
    def collect_problemId_objects(self,index):
        # df = self.kb["bongard"]
        # graph_id = df[df["problemId"] == index]
        graph_id = index
        triangles = self.kb["triangle"][self.kb["triangle"]["problemId"] == graph_id]
        squares = self.kb["square"][self.kb["square"]["problemId"] == graph_id]
        circles = self.kb["circle"][self.kb["circle"]["problemId"] == graph_id]
        ins = self.kb["in"][self.kb["in"]["problemId"] == graph_id]
        return triangles,squares,circles,ins
    
    def get_current_objects(self,index):
        # also removes the problem key column from the dataframe
        current_kb = {}
        for key in self.kb.keys():
            if key != self.dataset_name:
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
                current_kb[key].drop(columns=[self.problem_key],inplace=True)
            if key == self.dataset_name:
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
                current_kb[key].drop(columns=[self.target,self.problem_key],inplace=True)
        return current_kb
    
    # CHECKED 
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            # objectIds = np.unique(objectIds)

            # assumption that every table has an problemId column
            num_nodes = 0
            for key in current_kb.keys():
                num_nodes += len(current_kb[key])

            # create node feature array
            num_node_features = 8
            node_features = np.zeros((num_nodes, num_node_features))

            # create edge indices array : fully bidirectionally connected graph
            amount_of_edges = (num_nodes * (num_nodes - 1))
            edge_index = np.zeros((2,amount_of_edges))

            # create edge feature array
            edge_features = np.ones((amount_of_edges,1))

            # create node features
            total_index = 0
            for key in current_kb.keys():
                for index,row in current_kb[key].iterrows():
                    if key == "mutag":
                        node_features[total_index][0] = row["ind1"]
                        node_features[total_index][1] = row["inda"]
                        node_features[total_index][2] = row["logp"]
                        node_features[total_index][3] = row["lumo"]
                    if key == "atom":
                        node_features[total_index][4] = row["charge"]
                    if key == "bond":
                        node_features[total_index][5] = row["bond_type"]
                    if key == "nitro":
                        node_features[total_index][6] = 1
                    if key == "benzene":
                        node_features[total_index][7] = 1
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
        edge_predicates = ["bond","nitro","benzene"]
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 2
            num_nodes = len(current_kb["atom"])
            
            node_features = np.zeros((num_nodes, num_of_node_features))

            atom_type = {"c": 0, "h": 1, "o": 2, "n": 3, "s": 4, "cl": 5, "br": 6,"f":7,"i":8}

            node_mapping = {}
            index = 0
            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][0] = atom_type[row["atom_type"]]
                node_features[index][1] = row["charge"]
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
                        edge_feat = [0,0,0]
                        edge_feat[0] = row["bond_type"] 
                        if current_kb["nitro"].isin([row["atom_id_1"]]).any().any() and current_kb["nitro"].isin([row["atom_id_2"]]).any().any():
                            edge_feat[1] = 1
                        if current_kb["benzene"].isin([row["atom_id_1"]]).any().any() and current_kb["benzene"].isin([row["atom_id_2"]]).any().any():
                            edge_feat[2] = 1
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
            num_node_features = 2 # 1 for the atom fact and 1 for the instances

            # external and internal predicates
            num_edge_features = 4 # 1 for external and 1 for internal relations and then 2 more for nitro and benzene
            
            num_nodes = 1 + len(current_kb["atom"])
            # create the node features 
            node_features = np.zeros((num_nodes, num_node_features))

            node_features[0][0] = 1
            for i in range(1,len(current_kb["atom"]) + 1):
                node_features[i][1] = 1

            # create the edges
            object_indices = {"atom": 0}
            edge_indices = [[],[]]
            edge_features = []

            # Fill in external edges and the node feature
            
            node_mapping = {}
            index = 0
            for _,row in current_kb["atom"].iterrows():
                current_index = index + 1
                node_features[current_index][1] = 1
                node_mapping[row["atom_id"]] = current_index
                self.add_edges(edge_indices,0,current_index)
                edge_features.append([1,0,0,0,0])
                edge_features.append([1,0,0,0,0])
                index += 1
                
            
            # fill in the internal edges
            for index,row in current_kb["bond"].iterrows():
                node_index1 = node_mapping[row["atom_id_1"]]
                node_index2 = node_mapping[row["atom_id_2"]]
                edge_feat = [0,1,0,0,0]
                self.add_edges(edge_indices,node_index1,node_index2)
                edge_feat[2] = row["bond_type"]
                if current_kb["nitro"].isin([row["atom_id_1"]]).any().any() and current_kb["nitro"].isin([row["atom_id_2"]]).any().any():
                    edge_feat[3] = 1
                if current_kb["benzene"].isin([row["atom_id_1"]]).any().any() and current_kb["benzene"].isin([row["atom_id_2"]]).any().any():
                    edge_feat[4] = 1
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

            num_node_features = 5
            num_nodes = len(current_kb["atom"]) + len(current_kb["bond"]) + len(current_kb["nitro"]) + len(current_kb["benzene"])
            node_features = np.zeros((num_nodes, num_node_features))

            edge_indices = [[],[]]
            edge_features = []
            
            # create the node features from "atom"
            node_mapping = {}
            index = 0

            atom_type = {"c": 0, "h": 1, "o": 2, "n": 3, "s": 4, "cl": 5, "br": 6,"f":7,"i":8}
            for _,row in current_kb["atom"].iterrows():
                node_mapping[row["atom_id"]] = index
                node_features[index][0] = atom_type[row["atom_type"]]
                node_features[index][1] = row["charge"]
                index += 1
            
            for _,row in current_kb["bond"].iterrows():
                node_features[index][2] = 1
                node_id1 = node_mapping[row["atom_id_1"]]
                node_id2 = node_mapping[row["atom_id_2"]]
                self.add_edges(edge_indices,node_id1,index)
                edge_feat = [0,0]
                # check if row["atom_id_1"] and row["atom_id_2"] are in nitro or benzene 
                if current_kb["nitro"].isin([row["atom_id_1"]]).any().any():
                    edge_feat[0] = 1
                if current_kb["benzene"].isin([row["atom_id_1"]]).any().any():
                    edge_feat[1] = 1
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                self.add_edges(edge_indices,node_id2,index)
                # check if row["atom_id_1"] and row["atom_id_2"] are in nitro or benzene 
                if current_kb["nitro"].isin([row["atom_id_2"]]).any().any():
                    edge_feat[0] = 1
                if current_kb["benzene"].isin([row["atom_id_2"]]).any().any():
                    edge_feat[1] = 1
                edge_features.append(edge_feat)
                edge_features.append(edge_feat) 
                index += 1
            
            for _,row in current_kb["nitro"].iterrows():
                node_features[index][3] = 1
                node_id_1 = node_mapping[row["atom_id_1"]]
                node_id_2 = node_mapping[row["atom_id_2"]]
                node_id_3 = node_mapping[row["atom_id_3"]]
                node_id_4 = node_mapping[row["atom_id_4"]]
                self.add_edges(edge_indices,node_id_1,index)
                self.add_edges(edge_indices,node_id_2,index)
                self.add_edges(edge_indices,node_id_3,index)
                self.add_edges(edge_indices,node_id_4,index)
                index += 1
            
            for _,row in current_kb["benzene"].iterrows():
                node_features[index][4] = 1
                node_id_1 = node_mapping[row["atom_id_1"]]
                node_id_2 = node_mapping[row["atom_id_2"]]
                node_id_3 = node_mapping[row["atom_id_3"]]
                node_id_4 = node_mapping[row["atom_id_4"]]
                node_id_5 = node_mapping[row["atom_id_5"]]
                node_id_6 = node_mapping[row["atom_id_6"]]
                self.add_edges(edge_indices,node_id_1,index)
                self.add_edges(edge_indices,node_id_2,index)
                self.add_edges(edge_indices,node_id_3,index)
                self.add_edges(edge_indices,node_id_4,index)
                self.add_edges(edge_indices,node_id_5,index)
                self.add_edges(edge_indices,node_id_6,index)
                index += 1
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(np.ones((len(edge_indices[0]),1)), dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)
        
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=truth_label))

        return data_list
            
    def VirtualNode(self):
        edge_attr_emb = {
            1 : torch.tensor([1,0,0,0,0,0],dtype = torch.float),
            2 : torch.tensor([0,1,0,0,0,0],dtype = torch.float),
            3 : torch.tensor([0,0,1,0,0,0],dtype = torch.float),
            4 : torch.tensor([0,0,0,1,0,0],dtype = torch.float),
            5 : torch.tensor([0,0,0,0,1,0],dtype = torch.float),
            7 : torch.tensor([0,0,0,0,0,1],dtype = torch.float)
        }
        node_emb = {
            'c' : torch.tensor([1,0,0,0,0,0,0,0],dtype=torch.float),
            'n' : torch.tensor([0,1,0,0,0,0,0,0],dtype=torch.float),
            'o' : torch.tensor([0,0,1,0,0,0,0,0],dtype=torch.float),
            'h' : torch.tensor([0,0,0,1,0,0,0,0],dtype=torch.float),
            'cl': torch.tensor([0,0,0,0,1,0,0,0],dtype=torch.float),
            'f' : torch.tensor([0,0,0,0,0,1,0,0],dtype=torch.float),
            'br': torch.tensor([0,0,0,0,0,0,1,0],dtype=torch.float),
            'i' : torch.tensor([0,0,0,0,0,0,0,1],dtype=torch.float)
        }
        data_list = []

        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            atom_id_to_index = {}

            edge_indices = [[],[]]
            edge_attributes = []
            x = torch.empty((len(current_kb["atom"])+1,12), dtype=torch.float)

            # create nodes
            index = 0
            for _,row in current_kb["atom"].iterrows():
                atom_id_to_index[row['atom_id']] = index
                x[index] = torch.cat((node_emb[row['atom_type']],torch.zeros(4,dtype=torch.float)))
                index += 1
            
            # create edges and node attributes
            for _,row in current_kb["bond"].iterrows():
                edge_attr = edge_attr_emb[row['bond_type']]
                atom_id_1 = row['atom_id_1']
                index_1 = atom_id_to_index[atom_id_1]
                atom_id_2 = row['atom_id_2']
                index_2 = atom_id_to_index[atom_id_2]
                edge_indices[0].append(index_1)
                edge_indices[1].append(index_2)
                edge_attributes.append(edge_attr)
                edge_indices[0].append(index_2)
                edge_indices[1].append(index_1)
                edge_attributes.append(edge_attr)
            
            # add the graph info to the x
            drugs = current_kb["mutag"]
            ind1 = drugs['ind1'].iloc[0]
            inda = drugs['inda'].iloc[0]
            logp = drugs['logp'].iloc[0]
            lumo = drugs['lumo'].iloc[0]
            x[-1] = torch.concatenate((torch.zeros(8),torch.tensor([ind1,inda,logp,lumo],dtype=torch.float)))
            # add the indices
            for j in range(x.shape[0]-1):
                edge_indices[0].append(j)
                edge_indices[1].append(x.shape[0]-1)
                edge_attributes.append(torch.tensor([0,0,0,0,0,0],dtype=torch.float))
                edge_indices[0].append(x.shape[0]-1)
                edge_indices[1].append(j)
                edge_attributes.append(torch.tensor([0,0,0,0,0,0],dtype=torch.float))
            
            edge_indices = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attris = torch.empty((len(edge_attributes),6),dtype=torch.float)
            for i in range(len(edge_attributes)):
                edge_attris[i] = edge_attributes[i]
            
            y = torch.tensor(self.y[graph_index],dtype=torch.int64)
            
            graph = Data(x=x,edge_index=edge_indices,edge_attr=edge_attris,y=y)
            data_list.append(graph)
        return data_list