import torch
import pandas as pd
import glob
import os

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
        return f"{self.dataset_name}({problemId},{label}).\n"
    
    def node_edge(self,graphs,output_path):
        examples = []
        for i, graph in enumerate(graphs):

            example = ""
            problemId = str(i)
            example += self.truth_label(problemId,graph)

            for j,node in enumerate(graph.x):
                if node[0] == 1:
                    example += f"node({problemId},{j},red).\n"
                elif node[1] == 1:
                    example += f"node({problemId},{j},green).\n"
                elif node[2] == 1:
                    example += f"node({problemId},{j},blue).\n"
                elif node[3] == 1:
                    example += f"node({problemId},{j},yellow).\n"
            

            for i,edge in enumerate(graph.edge_index.T):
                example += f"edge({problemId},{edge[0]},{edge[1]}).\n"
                    
            
            examples.append(example)
        
        with open(output_path, "w") as f:
            for example in examples:
                f.write(example)
        return