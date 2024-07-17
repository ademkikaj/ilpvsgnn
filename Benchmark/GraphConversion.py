import glob
import pandas as pd
import os


class GraphConversion:

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
        # for key in kb.keys():
        #     if self.problem_key in kb[key].columns:
        #         kb[key] = kb[key].sort_values(by=[self.problem_key])
        return kb
    
    def create_y(self, kb):
        y = kb[self.dataset_name][self.target].to_numpy()
        y[y == "neg"] = 0
        y[y == "pos"] = 1
        y[y == " neg"] = 0
        y[y == " pos"] = 1
        return y

    def add_edges(self,edge_indices,node_index1,node_index2,edge_features=None,edge_feature=None):
        edge_indices[0].append(node_index1)
        edge_indices[1].append(node_index2)
        edge_indices[0].append(node_index2)
        edge_indices[1].append(node_index1)
        if edge_features is not None:
            edge_features.append(edge_feature)
            edge_features.append(edge_feature)
    
    def get_current_objects(self,index):
        # also removes the problem key column from the dataframe
        current_kb = {}
        for key in self.kb.keys():
            if key != self.dataset_name:
                if self.problem_key in self.kb[key].columns:
                    current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
                    #current_kb[key].drop(columns=[self.problem_key],inplace=True)
                else:
                    current_kb[key] = self.kb[key]
            if key == self.dataset_name:
                current_kb[key] = self.kb[key][self.kb[key][self.problem_key] == index]
                #current_kb[key].drop(columns=[self.target,self.problem_key],inplace=True)
        return current_kb

        
