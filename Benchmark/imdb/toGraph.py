import pandas as pd
import torch
import os
import glob
import torch_geometric
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.data import HeteroData
import re

class toGraph:
    def __init__(self, relational_path, dataset_name, dataset_problem_key,target, movie_encoder, genre_encoder, name_encoder):
        self.relational_path = relational_path
        self.dataset_name = dataset_name
        self.problem_key = dataset_problem_key
        self.target = target
        self.kb = self.collect_relational()
        self.num_of_graphs = len(self.kb[self.dataset_name][self.problem_key].unique())
        self.num_node_features = 3
        self.y = self.create_y(self.kb)

        self.movie_encoding = movie_encoder
        self.genre_encoding = genre_encoder
        self.name_encoding = name_encoder

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
        current_kb = {}
        current_kb[self.dataset_name] = self.kb[self.dataset_name].iloc[index]
        person1 = current_kb[self.dataset_name]["person1"]
        person2 = current_kb[self.dataset_name]["person2"]

        for key in self.kb.keys():
            if key in ["movie","genre","gender"]:
                current_kb[key] = self.kb[key][self.kb[key]["person"].isin([person1, person2])]
            if key in ["director","actor"]:
                current_kb[key] = self.kb[key][self.kb[key]["name"].isin([person1,person2])]
        return current_kb
    
    # CHECKED 
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            current_kb = self.get_current_objects(graph_index)
            
            # create node features
            num_node_features = len(self.name_encoding) + len(self.movie_encoding) + len(self.genre_encoding) + 1 + 1 + 1 + 1
            all_movies = list(current_kb["movie"]["movie"].unique())
            all_names = []
            all_names += list(current_kb["movie"]["person"].unique())
            all_names += list(current_kb["genre"]["person"].unique())
            all_names += list(current_kb["gender"]["person"].unique())
            all_names += list(current_kb["director"]["name"].unique())
            all_names += list(current_kb["actor"]["name"].unique())
            all_names = list(set(all_names))

            num_nodes =len(all_names)

            node_features = np.zeros((num_nodes, num_node_features))

            movie_start = len(self.name_encoding)
            movie_end = len(self.name_encoding) + len(self.movie_encoding)
            genre_start = len(self.name_encoding) + len(self.movie_encoding)
            genre_end = len(self.name_encoding) + len(self.movie_encoding) + len(self.genre_encoding)

            index = 0
            # add the person nodes:
            for name in all_names:
                node_features[index][0:len(self.name_encoding)] = self.name_encoding[name]
                if name in list(current_kb["movie"]["person"]):
                    node_features[index][len(self.name_encoding):len(self.name_encoding)+len(self.movie_encoding)] = self.movie_encoding[current_kb["movie"][current_kb["movie"]["person"] == name]["movie"].values[0]]
                if name in list(current_kb["genre"]["person"]):
                    node_features[index][len(self.name_encoding)+len(self.movie_encoding):len(self.name_encoding)+len(self.movie_encoding)+len(self.genre_encoding)] = self.genre_encoding[current_kb["genre"][current_kb["genre"]["person"] == name]["genre"].values[0]]
                if name in list(current_kb["gender"]["person"]):
                    if current_kb["gender"][current_kb["gender"]["person"] == name]["gender"].all() == "male":
                        node_features[index][genre_end] = 1
                    else:
                        node_features[index][genre_end+1] = 1
                if name in list(current_kb["director"]["name"]):
                    node_features[index][genre_end+2] = 1
                if name in list(current_kb["actor"]["name"]):
                    node_features[index][genre_end+3] = 1
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
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            current_kb = self.get_current_objects(graph_index)
            
            # create node features
            num_node_features = len(self.name_encoding) + len(self.movie_encoding)
            all_movies = list(current_kb["movie"]["movie"].unique())
            all_names = []
            all_names += list(current_kb["movie"]["person"].unique())
            all_names += list(current_kb["genre"]["person"].unique())
            all_names += list(current_kb["gender"]["person"].unique())
            all_names += list(current_kb["director"]["name"].unique())
            all_names += list(current_kb["actor"]["name"].unique())
            all_names = list(set(all_names))

            num_nodes =len(all_names) + len(all_movies)

            node_features = np.zeros((num_nodes, num_node_features))

            movie_start = len(self.name_encoding)
            movie_end = len(self.name_encoding) + len(self.movie_encoding)

            edge_indices = [[],[]]
            edge_features = []            

            index = 0
            node_mapping = {}
            # add the person nodes:
            for name in all_names:
                node_features[index][0:len(self.name_encoding)] = self.name_encoding[name]
                node_mapping[name] = index
                index += 1
            for movie in all_movies:
                node_features[index][movie_start:movie_end] = self.movie_encoding[movie]
                node_mapping[movie] = index
                index += 1
            
            for _,row in current_kb["movie"].iterrows():
                person_index = node_mapping[row["person"]]
                movie_index = node_mapping[row["movie"]]
                if row["person"] in list(current_kb["director"]["name"]):
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([0,1,0])
                    edge_features.append([0,1,0])
                elif row["person"] in list(current_kb["actor"]["name"]):
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([0,0,1])
                    edge_features.append([0,0,1])
                else:
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([1,0,0])
                    edge_features.append([1,0,0])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list
    
    def edge_based(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            current_kb = self.get_current_objects(graph_index)
            
            # create node features
            num_node_features = len(self.name_encoding) + len(self.movie_encoding) + 1
            all_movies = list(current_kb["movie"]["movie"].unique())
            all_names = []
            all_names += list(current_kb["movie"]["person"].unique())
            all_names += list(current_kb["genre"]["person"].unique())
            all_names += list(current_kb["gender"]["person"].unique())
            all_names += list(current_kb["director"]["name"].unique())
            all_names += list(current_kb["actor"]["name"].unique())
            all_names = list(set(all_names))

            num_nodes = 2 * (len(all_names) + len(all_movies))

            node_features = np.zeros((num_nodes, num_node_features))

            movie_start = len(self.name_encoding)
            movie_end = len(self.name_encoding) + len(self.movie_encoding)

            edge_indices = [[],[]]
            edge_features = []            

            index = 0
            node_mapping = {}
            # add the person nodes:
            for name in all_names:
                node_features[index][0:len(self.name_encoding)] = self.name_encoding[name]
                index += 1
                node_features[index+1][-1] = 1
                self.add_edges(edge_indices,index,index+1)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
                node_mapping[name] = index
            for movie in all_movies:
                node_features[index][movie_start:movie_end] = self.movie_encoding[movie]
                index += 1
                node_features[index+1][-1] = 1
                self.add_edges(edge_indices,index,index+1)
                edge_features.append([1,0,0,0])
                edge_features.append([1,0,0,0])
                node_mapping[movie] = index
                index += 1
            
            for _,row in current_kb["movie"].iterrows():
                person_index = node_mapping[row["person"]]
                movie_index = node_mapping[row["movie"]]
                if row["person"] in list(current_kb["director"]["name"]):
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([0,1,0,0])
                    edge_features.append([0,1,0,0])
                elif row["person"] in list(current_kb["actor"]["name"]):
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([0,0,1,0])
                    edge_features.append([0,0,1,0])
                else:
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([0,0,0,1])
                    edge_features.append([0,0,0,1])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list

    def Klog(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            current_kb = self.get_current_objects(graph_index)
            
            # create node features
            num_node_features = len(self.name_encoding) + len(self.movie_encoding) + 2
            all_movies = list(current_kb["movie"]["movie"].unique())
            all_names = []
            all_names += list(current_kb["movie"]["person"].unique())
            all_names += list(current_kb["genre"]["person"].unique())
            all_names += list(current_kb["gender"]["person"].unique())
            all_names += list(current_kb["director"]["name"].unique())
            all_names += list(current_kb["actor"]["name"].unique())
            all_names = list(set(all_names))

            num_nodes =len(all_names) + len(all_movies) + len(current_kb["director"]) + len(current_kb["actor"])

            node_features = np.zeros((num_nodes, num_node_features))

            movie_start = len(self.name_encoding)
            movie_end = len(self.name_encoding) + len(self.movie_encoding)

            edge_indices = [[],[]]
            edge_features = []            

            index = 0
            node_mapping = {}
            # add the person nodes:
            for name in all_names:
                node_features[index][0:len(self.name_encoding)] = self.name_encoding[name]
                node_mapping[name] = index
                index += 1
            for movie in all_movies:
                node_features[index][movie_start:movie_end] = self.movie_encoding[movie]
                node_mapping[movie] = index
                index += 1
            
            for _,row in current_kb["movie"].iterrows():
                person_index = node_mapping[row["person"]]
                movie_index = node_mapping[row["movie"]]
                if row["person"] in list(current_kb["director"]["name"]):
                    node_features[index][-2] = 1
                    self.add_edges(edge_indices,person_index,index)
                    edge_features.append([0,1,0])
                    edge_features.append([0,1,0])
                    self.add_edges(edge_indices,index,movie_index)
                    edge_features.append([0,1,0])
                    edge_features.append([0,1,0])
                    index += 1
                elif row["person"] in list(current_kb["actor"]["name"]):
                    node_features[index][-1] = 1
                    self.add_edges(edge_indices,person_index,index)
                    edge_features.append([0,0,1])
                    edge_features.append([0,0,1])
                    self.add_edges(edge_indices,index,movie_index)
                    edge_features.append([0,0,1])
                    edge_features.append([0,0,1])
                    index += 1
                else:
                    self.add_edges(edge_indices,person_index,movie_index)
                    edge_features.append([1,0,0])
                    edge_features.append([1,0,0])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(self.y[graph_index], dtype=torch.int64)))
        return data_list

    
