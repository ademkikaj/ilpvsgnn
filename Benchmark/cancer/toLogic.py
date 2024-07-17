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
        self.decoders = self.load_decoders()

    def load_decoders(self):
        with open("Benchmark/financial/decoders.json") as f:
            decoders = json.load(f)
        return decoders
    
    def truth_label(self,problemId,graph):
        if graph.y.item() == 1:
            label = "pos"
        else:
            label = "neg"
        return f"{self.dataset_name}({problemId},{label}).\n"
        
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
    
    def get_current_objects(self, index):
        current_objects = {}
        for key in self.kb.keys():
            if "id" in self.kb[key].columns:
                if index in self.kb[key]["id"].values:
                    current_objects[key] = self.kb[key][self.kb[key]["id"] == index]
                else:
                    current_objects[key] = pd.DataFrame()
        return current_objects

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}
    
    def node_only(self,graphs,output_path):
        pass
    
    def node_edge(self,graphs,output_path):
        # assumption that the relational path given is training or testing, and equivalent with the input graphs
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = self.kb[self.dataset_name][self.problem_key][i]
            current_kb = self.get_current_objects(problem_id)

            assert len(graph.x) == len(current_kb["atom"]) 
    
            # add the class example and the truth label
            example += self.truth_label(problem_id,graph)

            # add all atom relations, loop over the tables 
            for _,row in current_kb["atom"].iterrows():
                relation = "atom"
                args = []
                for _,col in enumerate(current_kb[relation].columns):
                    args.append(str(row[col]))
                example += f"{relation}({','.join(args)}).\n"
            for _,row in current_kb["bond"].iterrows():
                relation = "bond"
                args = []
                for _,col in enumerate(current_kb[relation].columns):
                    args.append(str(row[col]))
                example += f"{relation}({','.join(args)}).\n"
            
            examples.append(example)
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def edge_based(self,graphs,output_path):
        pass

    def Klog(self,graphs,output_path):
        pass