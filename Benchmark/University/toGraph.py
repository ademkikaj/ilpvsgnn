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
        for key in kb.keys():
            if key in kb[key].columns:
                kb[key] = kb[key].sort_values(by=[self.problem_key])
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
            if key == "university" or key == "registration" or key == "ra":
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
        for key in self.kb.keys():
            if key == "course":
                # go over all registered and get the courseId
                course_ids = []
                for _,row in current_kb["registration"].iterrows():
                    course_ids.append(row["course_id"])
                current_kb[key] = self.kb[key][self.kb[key]["course_id"].isin(course_ids)]
            if key == "prof":
                prof_ids = []
                for _,row in current_kb["ra"].iterrows():
                    prof_ids.append(row["prof_id"])
                current_kb[key] = self.kb[key][self.kb[key]["prof_id"].isin(prof_ids)]
        return current_kb
    
    # CHECKED 
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)
            
            # create node features
            num_node_features = 11
            num_nodes = len(current_kb["university"]) + len(current_kb["course"]) + len(current_kb["prof"]) + len(current_kb["ra"]) + len(current_kb["registration"])

            node_features = np.zeros((num_nodes, num_node_features))

            salary_mapping = {"low":0,"med":1,"high":2}

            index = 0
            # add the student features
            for _,row in current_kb["university"].iterrows():
                node_features[index][0] = row["ranking"]
                index += 1
            
            # add the course nodes
            courseIds = {}
            for _,row in current_kb["course"].iterrows():
                node_features[index][1] = row["diff"]
                node_features[index][2] = row["rating"]
                courseIds[row["course_id"]] = index
                index += 1
            
            # add the prof nodes
            profIds = {}
            for _,row in current_kb["prof"].iterrows():
                node_features[index][3] = row["teachingability"]
                node_features[index][4] = row["popularity"]
                profIds[row["prof_id"]] = index
                index += 1
            # add the registration nodes
            for _,row in current_kb["registration"].iterrows():
                node_features[index][5] = courseIds[row["course_id"]]
                node_features[index][6] = row["grade"]
                node_features[index][7] = row["sat"]
                index += 1
            # add the RA nodes
            for _,row in current_kb["ra"].iterrows():
                node_features[index][8] = profIds[row["prof_id"]]
                node_features[index][9] = salary_mapping[row["salary"]]
                node_features[index][10] = row["capability"]
                index += 1
            
            edge_indices = [[],[]]
            # fully connect the edges
            for i in range(num_nodes):
                for j in range(num_nodes):
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
        salary_mapping = {"low":0,"med":1,"high":2}
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 5
            num_nodes = len(current_kb["university"]) + len(current_kb["course"]) + len(current_kb["prof"]) 

            node_features = np.zeros((num_nodes, num_of_node_features))

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
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            # add edges
            for _,row in current_kb["registration"].iterrows():
                student_index = node_mapping[row["student_id"]]
                course_index = node_mapping[row["course_id"]]
                self.add_edges(edge_indices,student_index,course_index)
                edge_features.append([row["grade"],row["sat"],0,0])
                edge_features.append([row["grade"],row["sat"],0,0])
            for _,row in current_kb["ra"].iterrows():
                prof_index = node_mapping[row["prof_id"]]
                student_index = node_mapping[row["student_id"]]
                self.add_edges(edge_indices,prof_index,student_index)
                edge_features.append([0,0,salary_mapping[row["salary"]],row["capability"]])
                edge_features.append([0,0,salary_mapping[row["salary"]],row["capability"]])

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = torch.tensor(edge_indices, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list
    
    def edge_based(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features
        salary_mapping = {"low":0,"med":1,"high":2}
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 5
            num_nodes = len(current_kb["university"]) + len(current_kb["course"]) + len(current_kb["prof"]) 

            node_features = np.zeros((num_nodes, num_of_node_features))

            # add nodes
            node_mapping = {}
            index = 0
            for _,row in current_kb["university"].iterrows():
                node_features[index][0] = row["ranking"]
                node_mapping[row["student_id"]] = index
                index += 1
            for _,row in current_kb["course"].iterrows():
                node_features[index][1] = row["diff"]
                node_mapping[row["course_id"]] = index
                index += 1
            for _,row in current_kb["prof"].iterrows():
                node_features[index][3] = row["teachingability"]
                node_mapping[row["prof_id"]] = index
                index += 1
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            # add edges
            for _,row in current_kb["registration"].iterrows():
                student_index = node_mapping[row["student_id"]]
                course_index = node_mapping[row["course_id"]]
                self.add_edges(edge_indices,student_index,course_index)
                edge_features.append([row["grade"],row["sat"],0,0])
                edge_features.append([row["grade"],row["sat"],0,0])
            for _,row in current_kb["ra"].iterrows():
                prof_index = node_mapping[row["prof_id"]]
                student_index = node_mapping[row["student_id"]]
                self.add_edges(edge_indices,prof_index,student_index)
                edge_features.append([0,0,salary_mapping[row["salary"]],row["capability"]])
                edge_features.append([0,0,salary_mapping[row["salary"]],row["capability"]])

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = torch.tensor(edge_indices, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
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
            num_nodes = len(current_kb["university"]) + len(current_kb["course"]) + len(current_kb["prof"]) + len(current_kb["ra"]) + len(current_kb["registration"])

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
            for _,row in current_kb["ra"].iterrows():
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

    
