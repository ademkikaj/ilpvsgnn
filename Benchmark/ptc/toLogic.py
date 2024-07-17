import torch
import pandas as pd
import glob
import os
import json

class toLogic:
    
    def __init__(self,dataset_name,relational_path,problem_key) -> None:
        self.dataset_name = dataset_name
        self.relational_path = relational_path
        self.problem_key = problem_key
        self.kb = self.collect_relational()
    
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
            if self.problem_key in kb[key].columns:
                kb[key] = kb[key].sort_values(by=[self.problem_key])
        return kb
    
    def truth_label(self,problemId,graph):
        if graph.y.item() == 1:
            label = "pos"
        else:
            label = "neg"
        problemId = int(problemId[2:])
        return f"{self.dataset_name}({problemId},{label}).\n"

    def get_current_objects(self, index):
        current_objects = {}
        for key in self.kb.keys():
            if "id" in self.kb[key].columns:
                if index in self.kb[key]["id"].values:
                    current_objects[key] = self.kb[key][self.kb[key]["id"] == index]
                else:
                    current_objects[key] = pd.DataFrame()
        return current_objects
    
    def node_only(self,graphs,output_path):
        pass
    
    def node_edge(self,graphs,output_path):
         
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problemId = self.kb[self.dataset_name][self.problem_key][i]
            current_kb = self.get_current_objects(problemId)
            # connected if atom_id is in atom atom_id and atom_id2 is in atom atom_id
            current_kb["connected"] = self.kb["connected"][self.kb["connected"]["atom_id"].isin(current_kb["atom"]["atom_id"]) & self.kb["connected"]["atom_id2"].isin(current_kb["atom"]["atom_id"])]

            assert len(graph.x) == len(current_kb["atom"])

            # add the class label
            example += self.truth_label(problemId,graph)

            # add all atom relations
            for _,row in current_kb["atom"].iterrows():
                # lets write the ids as ints
                row_id = int(row["id"][2:])
                row_atom_id = "n_"+str(int(row["atom_id"][2:]))
                row_element = row["element"]
                example += f"atom({row_id},{row_atom_id},{row_element}).\n"
            
            # add all bond relations
            for _,row in current_kb["connected"].iterrows():
                bond_type = self.kb["bond"][self.kb["bond"]["bond_id"] == row["bond_id"]]["bond_type"].values[0]
                row_id = int(problemId[2:])
                atom_id1 = "n_" + str(int(row["atom_id"][2:]))
                atom_id2 = "n_" + str(int(row["atom_id2"][2:]))
                example += f"connected({row_id},{atom_id1},{atom_id2},{bond_type}).\n"
            
            examples.append(example)

        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def edge_based(self,graphs,output_path):
        pass

    def Klog(self,graphs,output_path):
        pass