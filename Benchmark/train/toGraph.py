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
        #kb[self.dataset_name] = kb[self.dataset_name].sort_values(by=[self.problem_key])
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
    
    def get_current_objects(self,index):
        # also removes the problem key column from the dataframe
        current_kb = {}
        current_kb["train"] = self.kb["train"][self.kb["train"][self.problem_key] == index]
        current_kb["has_car"] = self.kb["has_car"][self.kb["has_car"][self.problem_key] == index]
        car_ids = current_kb["has_car"]["car_id"]
        current_kb["has_load"] = self.kb["has_load"][self.kb["has_load"]["car_id"].isin(car_ids)]
        load_ids = current_kb["has_load"]["load_id"]

        for key in self.kb.keys():
            if key not in ["train","has_car","has_load"]:
                if "car_id" in self.kb[key].columns:
                    current_kb[key] = self.kb[key][self.kb[key]["car_id"].isin(car_ids)]
                elif "load_id" in self.kb[key].columns:
                    current_kb[key] = self.kb[key][self.kb[key]["load_id"].isin(load_ids)]
        return current_kb


    def node_only(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features
        node_predicates = ["krk"]
        edge_predicates = ["bond","nitro","benzene"]
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 1 + 6 + 6
            num_nodes = len(current_kb["train"]) + len(current_kb["has_car"]) + len(current_kb["has_load"])

            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            edge_indices = [[],[]]
            edge_features = []
            index = 0
            for _,row in current_kb["train"].iterrows():
                node_features[index][0] = 1
                node_mapping[row[self.problem_key]] = index
                index += 1
            for _,row in current_kb["has_car"].iterrows():
                node_mapping[row["car_id"]] = index
                node_features[index][1] = current_kb["short"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][2] = current_kb["long"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][3] = current_kb["two_wheels"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][4] = current_kb["three_wheels"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][5] = current_kb["roof_open"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][6] = current_kb["roof_closed"]["car_id"].isin([row["car_id"]]).any()
                index += 1
            for _,row in current_kb["has_load"].iterrows():
                node_mapping[row["load_id"]] = index
                node_features[index][7] = current_kb["zero_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][8] = current_kb["one_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][9] = current_kb["two_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][10] = current_kb["three_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][11] = current_kb["circle"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][12] = current_kb["triangle"]["load_id"].isin([row["load_id"]]).any()
                index += 1        

            # fully connected edges
            for i in range(num_nodes):
                for j in range(i+1,num_nodes):
                    if i != j:
                        self.add_edges(edge_indices,i,j)
  

            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.ones(edge_index.shape[1],1)

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = edge_features,
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list
    

    def node_edge(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features

        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 1 + 6 + 6
            num_nodes = len(current_kb["train"]) + len(current_kb["has_car"]) + len(current_kb["has_load"])

            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            edge_indices = [[],[]]
            edge_features = []
            index = 0
            for _,row in current_kb["train"].iterrows():
                node_features[index][0] = 1
                node_mapping[row[self.problem_key]] = index
                index += 1
            for _,row in current_kb["has_car"].iterrows():
                node_mapping[row["car_id"]] = index
                node_features[index][1] = current_kb["short"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][2] = current_kb["long"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][3] = current_kb["two_wheels"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][4] = current_kb["three_wheels"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][5] = current_kb["roof_open"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][6] = current_kb["roof_closed"]["car_id"].isin([row["car_id"]]).any()
                # add edge between train and car
                self.add_edges(edge_indices,node_mapping[row[self.problem_key]],index)
                edge_features.append([1,0])
                edge_features.append([1,0])
                index += 1
            for _,row in current_kb["has_load"].iterrows():
                node_mapping[row["load_id"]] = index
                node_features[index][7] = current_kb["zero_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][8] = current_kb["one_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][9] = current_kb["two_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][10] = current_kb["three_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][11] = current_kb["circle"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][12] = current_kb["triangle"]["load_id"].isin([row["load_id"]]).any()
                # add edge between car and load
                self.add_edges(edge_indices,node_mapping[row["car_id"]],index)
                edge_features.append([0,1])
                edge_features.append([0,1])
                index += 1            

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
        # 
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 4 + 6 + 6
            num_nodes = len(current_kb["train"]) + len(current_kb["has_car"]) + len(current_kb["has_load"]) + 15

            node_features = np.zeros((num_nodes, num_of_node_features))

            node_features[0][1] = 1 # train
            node_features[1][2] = 1 # car  
            node_features[2][3] = 1 # load
            node_features[3][4] = 1 # short
            node_features[4][5] = 1 # long
            node_features[5][6] = 1 # two_wheels
            node_features[6][7] = 1 # three_wheels
            node_features[7][8] = 1 # roof_open
            node_features[8][9] = 1 # roof_closed
            node_features[9][10] = 1 # zero_load
            node_features[10][11] = 1 # one_load
            node_features[11][12] = 1 # two_load
            node_features[12][13] = 1 # three_load
            node_features[13][14] = 1 # circle
            node_features[14][15] = 1 # triangle

            node_mapping = {}
            edge_indices = [[],[]]
            edge_features = []
            index = 15
            for _,row in current_kb["train"].iterrows():
                node_features[index][0] = 1
                node_mapping[row["id"]] = index
                # add edge between train and train
                self.add_edges(edge_indices,index,0)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
                index += 1
            for _,row in current_kb["has_car"].iterrows():
                node_features[index][0] = 1
                node_mapping[row["car_id"]] = index
                # add edge between car and car
                self.add_edges(edge_indices,index,1)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
                # add edge between train and car external
                self.add_edges(edge_indices,node_mapping[row["id"]],index)
                edge_features.append([0,1,1,0])
                edge_features.append([0,1,1,0])

                # Add internal edges
                if current_kb["short"]["car_id"].isin([row["car_id"]]).any():
                    self.add_edges(edge_indices,index,3)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["long"]["car_id"].isin([row["car_id"]]).any():
                    self.add_edges(edge_indices,index,4)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["two_wheels"]["car_id"].isin([row["car_id"]]).any():
                    self.add_edges(edge_indices,index,5)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["three_wheels"]["car_id"].isin([row["car_id"]]).any():
                    self.add_edges(edge_indices,index,6)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["roof_open"]["car_id"].isin([row["car_id"]]).any():
                    self.add_edges(edge_indices,index,7)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["roof_closed"]["car_id"].isin([row["car_id"]]).any():
                    self.add_edges(edge_indices,index,8)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                index += 1
            for _,row in current_kb["has_load"].iterrows():
                node_features[index][0] = 1
                node_mapping[row["load_id"]] = index
                # add edge between load and load
                self.add_edges(edge_indices,index,2)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
                # add edge between car and load external
                self.add_edges(edge_indices,node_mapping[row["car_id"]],index)
                edge_features.append([0,1,0,1])
                edge_features.append([0,1,0,1])

                # Add internal edges
                if current_kb["zero_load"]["load_id"].isin([row["load_id"]]).any():
                    self.add_edges(edge_indices,index,9)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["one_load"]["load_id"].isin([row["load_id"]]).any():
                    self.add_edges(edge_indices,index,10)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["two_load"]["load_id"].isin([row["load_id"]]).any():
                    self.add_edges(edge_indices,index,11)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["three_load"]["load_id"].isin([row["load_id"]]).any():
                    self.add_edges(edge_indices,index,12)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["circle"]["load_id"].isin([row["load_id"]]).any():
                    self.add_edges(edge_indices,index,13)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                if current_kb["triangle"]["load_id"].isin([row["load_id"]]).any():
                    self.add_edges(edge_indices,index,14)
                    edge_features.append([1,0,0,0])
                    edge_features.append([1,0,0,0])
                index += 1            

            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = edge_features,
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list

    def Klog(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features

        node_predicates = ["krk"]
        edge_predicates = ["bond","nitro","benzene"]
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 1 + 2 + 6 + 6 
            num_nodes = len(current_kb["train"]) + 2*len(current_kb["has_car"]) + 2*len(current_kb["has_load"])

            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            edge_indices = [[],[]]
            index = 0
            for _,row in current_kb["train"].iterrows():
                node_features[index][0] = 1
                node_mapping[row["id"]] = index
                index += 1
            for _,row in current_kb["has_car"].iterrows():
                node_mapping[row["car_id"]] = index
                node_features[index][1] = current_kb["short"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][2] = current_kb["long"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][3] = current_kb["two_wheels"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][4] = current_kb["three_wheels"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][5] = current_kb["roof_open"]["car_id"].isin([row["car_id"]]).any()
                node_features[index][6] = current_kb["roof_closed"]["car_id"].isin([row["car_id"]]).any()
                # add edge between train and car
                car_index = index
                train_index = node_mapping[row["id"]]
                # new klog node
                node_features[index + 1][13] = 1
                # add edges between train and klog node
                self.add_edges(edge_indices,train_index,index+1)
                # add edges between klog node and car
                self.add_edges(edge_indices,index +1,car_index)

                index += 2
            for _,row in current_kb["has_load"].iterrows():
                node_mapping[row["load_id"]] = index
                node_features[index][7] = current_kb["zero_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][8] = current_kb["one_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][9] = current_kb["two_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][10] = current_kb["three_load"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][11] = current_kb["circle"]["load_id"].isin([row["load_id"]]).any()
                node_features[index][12] = current_kb["triangle"]["load_id"].isin([row["load_id"]]).any()
                # add edge between car and load
                load_index = index
                car_index = node_mapping[row["car_id"]]
                # new klog node
                node_features[index + 1][14] = 1
                # add edges between car and klog node
                self.add_edges(edge_indices,car_index,index+1)
                # add edges between klog node and load
                self.add_edges(edge_indices,index +1,load_index)
                index += 2

            edge_index = torch.tensor(edge_indices, dtype=torch.int64)


            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = torch.ones(edge_index.shape[1],1),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list
