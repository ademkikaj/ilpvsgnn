from torch_geometric.data import Data
import torch
import numpy as np
import pandas as pd
import os
import glob

# generalised conversions of relational data to graph data

class toGraph:

    def __init__(self,relational_path,dataset_name,node_predicates,edge_predicates,problem_key="problemId"):
        self.relational_path = relational_path
        self.dataset_name = dataset_name
        self.problem_key = problem_key
        self.kb = self.collect_relational_data()
        self.y = self.create_y()
    
        self.tables = list(self.kb.keys()).remove(self.dataset_name)

        self.num_graphs = len(self.kb[self.dataset_name][self.problem_key].unique())

        # for node_edge representation
        self.node_predicates = node_predicates
        self.egde_predicates = edge_predicates
    
    def collect_relational_data(self):
        kb = {}
        predicates = []
        all_files = glob.glob(os.path.join(self.relational_path, "*.csv"))
        for filename in all_files:
            df = pd.read_csv(filename)
            relation = filename.split('/')[-1].split('.')[0]
            predicates.append(relation)
            kb[relation] = df
        # sort all the dataframes??
        for key in kb.keys():
            kb[key] = kb[key].sort_values(by=[self.problem_key])    
        return kb
    
    def create_y(self):
        y = self.kb[self.dataset_name]["class"].to_numpy()
        y[y == "neg"] = 0
        y[y == " neg"] = 0
        y[y == "pos"] = 1
        y[y == " pos"] = 1
        return y
    
    def add_edges(self,edge_indices,node_index1,node_index2,edge_index):
        edge_indices[0].append(node_index1)
        edge_indices[1].append(node_index2)
        edge_index += 1
        edge_indices[0].append(node_index2)
        edge_indices[1].append(node_index1)
        edge_index += 1
    
    def collect_objects(self,index):
        result = {}
        for table in self.tables:
            result[table] = self.kb[table][self.kb[table][self.problem_key] == index]
        return result
    

    def node_only(self):
        data_list = []
        for graph_index in range(self.num_graphs):

            data = self.collect_objects(graph_index)

            # Determining the number of nodes: in this representation all tables are nodes
            num_nodes = 0
            # amount of node features is the amount of columns in the dataframe minus the problem key.
            num_node_features = 0

            for key in data.keys():
                num_node_features += len(data[key].columns) - 1
                num_nodes += len(data[key])
            
            node_features = np.zeros((num_nodes,num_node_features))

            # fill in the node features array
            node_index = 0
            for i,key in data.keys():
                for j in range(len(data[key])):
                    node_features[node_index][i] = 1
                    node_index += 1        

            # creating the edge indices array: fully bidirectionally connected graph
            amount_of_edges = (num_nodes * (num_nodes - 1))
            edge_index = np.zeros((2,amount_of_edges))

            # fill in the edge indices
            index = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index[0][index] = i
                        edge_index[1][index] = j
                        index += 1

            # create edge feature array
            edge_features = np.zeros((amount_of_edges,1))

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = torch.tensor(edge_index, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
                y = torch.tensor([self.y[graph_index]], dtype=torch.float)
            ))
        return data_list
    
    def node_and_edge(self):
        data_list = []
        for graph_index in range(self.num_graphs):

            data = self.collect_objects(graph_index)

            
            # num_node_features is the amount of node_predicates

        pass
