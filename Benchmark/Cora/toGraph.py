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


        self.paper_id_encoder = self.create_encoder("cora","paper_id")
        self.word_id_encoder = self.create_encoder("content","word_cited_id")

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
        for key in self.kb.keys():
            if key == self.dataset_name:
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
        for key in self.kb.keys():
            if key == "content":
                current_kb[key] = self.kb[key][self.kb[key]["paper_id"] == index]
            if key == "cites":
                current_kb[key] = self.kb[key][self.kb[key]["citing_paper_id"] == index]
        return current_kb

    def create_encoder(self,table,column):
        one_hot_encoding = {}
        categories = sorted(self.kb[table][column].unique())
        for i,category in enumerate(categories):
            encoding = np.zeros(len(categories))
            encoding[i] = 1
            one_hot_encoding[category] = encoding
        return one_hot_encoding
    
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)
            
            # create node features 
            num_nodes = len(current_kb["cora"]) + len(current_kb["content"]) + len(current_kb["cites"])
            num_node_features = len(self.paper_id_encoder) + len(self.word_id_encoder)
            node_features = np.zeros((num_nodes, num_node_features))

            len_paper_id = len(self.paper_id_encoder) 

            # add nodes
            # add the main paper node
            index = 0
            for _,row in current_kb["cora"].iterrows():
                node_features[index][:len_paper_id] = self.paper_id_encoder[row["paper_id"]]
                index += 1
            for _,row in current_kb["content"].iterrows():
                node_features[index][len_paper_id:] = self.word_id_encoder[row["word_cited_id"]]
                index += 1
            for _,row in current_kb["cites"].iterrows():
                node_features[index][:len_paper_id] = self.paper_id_encoder[row["cited_paper_id"]]
                index += 1

            # add the edges: fully connected
            edge_indices = [[],[]]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        self.add_edges(edge_indices,i,j)
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attr = torch.ones((edge_index[0].shape[0],1), dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label))
        return data_list

    def node_edge(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = len(self.paper_id_encoder) + len(self.word_id_encoder)
            num_nodes = len(current_kb["cora"]) + len(current_kb["content"]) + len(current_kb["cites"])

            node_features = np.zeros((num_nodes, num_of_node_features))

            edge_indices = [[],[]]
            edge_features = []

            len_paper_id = len(self.paper_id_encoder)

            # add nodes
            node_mapping = {}
            index = 0
            for _,row in current_kb["cora"].iterrows():
                node_features[index][:len_paper_id] = self.paper_id_encoder[row["paper_id"]]
                index += 1
            
            for _,row in current_kb["content"].iterrows():
                node_features[index][len_paper_id:] = self.word_id_encoder[row["word_cited_id"]]
                self.add_edges(edge_indices,index,0)
                index += 1

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = torch.tensor(edge_indices, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list
    
    def edge_based(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # 3 for the external predicates and 1 for instances
            num_node_features = 6 # 1 for the atom fact and 1 for the instances

            # external and internal predicates
            num_edge_features = 6 # 1 for external and 1 for internal relations and then 4 for the relation features
            
            num_nodes = 6
            # create the node features 
            node_features = []


            # first collect all the facts
            n_facts = 0
            student_facts = {}
            for _,row in current_kb["student"]:
                student_facts[row["ranking"]] = n_facts
                n_facts += 1
            course_diff_facts = {}
            course_rating_facts = {}
            for _,row in current_kb["course"]:
                course_diff_facts[row["diff"]] = n_facts
                n_facts += 1
                course_rating_facts[row["rating"]] = n_facts
                n_facts += 1
            prof_teachability_facts = {}
            prof_popularity_facts = {}
            for _,row in current_kb["prof"]:
                prof_teachability_facts[row["teachability"]] = n_facts
                n_facts += 1
                prof_popularity_facts[row["popularity"]] = n_facts
                n_facts += 1
            
            num_node_features = 5 + n_facts
            num_nodes = n_facts + len(current_kb["student"]) + len(current_kb["course"]) + len(current_kb["prof"])
            node_features = np.zeros((num_nodes, num_node_features))

            # add the nodes
            index = 0
            for key in student_facts.keys():
                node_features[index][5 + student_facts[key]] = 1
                index += 1
            
            edge_indices = [[],[]]
            edge_features = []
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label))

        return data_list

    def Klog(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features
        salary_mapping = {"low":0,"med":1,"high":2}
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 9
            num_nodes = len(current_kb["university"]) + len(current_kb["course"]) + len(current_kb["prof"]) + len(current_kb["RA"]) + len(current_kb["registration"])

            node_features = np.zeros((num_nodes, num_of_node_features))

            # create edge indices
            edge_indices = [[],[]]

            # add nodes
            node_mapping = {}
            index = 0
            for _,row in current_kb["university"].iterrows():
                node_features[index][0] = row["ranking"]
                node_mapping[row["student_id"]] = index
                index += 1
            for _,row in current_kb["course"].iterrows():
                node_features[index][1] = row["diff"]
                node_features[index][2] = row["rating"]
                node_mapping[row["course_id"]] = index
                index += 1
            for _,row in current_kb["prof"].iterrows():
                node_features[index][3] = row["teachingability"]
                node_features[index][4] = row["popularity"]
                node_mapping[row["prof_id"]] = index
                index += 1
            for _,row in current_kb["registration"].iterrows():
                node_features[index][5] = row["grade"]
                node_features[index][6] = row["sat"]
                # add the edges
                student_index = node_mapping[row["student_id"]]
                course_index = node_mapping[row["course_id"]]
                self.add_edges(edge_indices,student_index,course_index)
                index += 1
            for _,row in current_kb["RA"].iterrows():
                node_features[index][7] = salary_mapping[row["salary"]]
                node_features[index][8] = row["capability"]
                # add the edges
                prof_index = node_mapping[row["prof_id"]]
                student_index = node_mapping[row["student_id"]]
                self.add_edges(edge_indices,prof_index,student_index)
                index += 1
            
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.ones((edge_index[0].shape[0],1), dtype=torch.float)

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = edge_features,
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list

    
