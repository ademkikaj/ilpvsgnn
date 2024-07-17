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
        return kb
    
    def truth_label(self,problemId,graph):
        if graph.y.item() == 1:
            label = "pos"
        else:
            label = "neg"
        return f"{self.dataset_name}({problemId},{label}).\n"
    
    def node_only(self,graphs,output_path):
        examples = []
        # add the colors
        colors = ""
        colors += "red(red).\n"
        colors += "green(green).\n"
        examples.append(colors)

        for i,graph in enumerate(graphs):
            example = ""
            problemId = self.kb[self.dataset_name][self.problem_key].unique()[i]
            spl = problemId.split("_")
            prefix = problemId[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]
            
            example += self.truth_label(problemId,graph)
            for j,node in enumerate(graph.x):
                if node[0] == 1:
                    example += f"node({prefix}{j},red).\n"
                elif node[1] == 1:
                    example += f"node({prefix}{j},green).\n"
                elif node[2] == 1:
                    example += f"node({problemId},{j},blue).\n"
                elif node[3] == 1:
                    example += f"node({problemId},{j},yellow).\n"
                

            for i,edge in enumerate(graph.edge_index.T):
                example += f"edge({edge[0]},{edge[1]}).\n"
            
            examples.append(example)
        
        with open(output_path, "w") as f:
            for example in examples:
                f.write(example)
        return 
    
    def node_edge(self,graphs,output_path):
        examples = []
        # add the colors
        colors = ""
        colors += "red(red).\n"
        colors += "green(green).\n"
        examples.append(colors)
        for i,graph in enumerate(graphs):
            example = ""
            problemId = self.kb[self.dataset_name][self.problem_key].unique()[i]
            spl = problemId.split("_")
            prefix = problemId[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]
            
            example += self.truth_label(problemId,graph)
            for j,node in enumerate(graph.x):
                if node[0] == 1:
                    example += f"node({prefix}{j},red).\n"
                elif node[1] == 1:
                    example += f"node({prefix}{j},green).\n"
                elif node[2] == 1:
                    example += f"node({problemId},{j},blue).\n"
                elif node[3] == 1:
                    example += f"node({problemId},{j},yellow).\n"
                elif node[4] == 1:
                    example += f"edge_node({problemId},{j}).\n"

            for i,edge in enumerate(graph.edge_index.T):
                example += f"edge({prefix}{edge[0]},{prefix}{edge[1]}).\n"
            
            examples.append(example)
        
        with open(output_path, "w") as f:
            for example in examples:
                f.write(example)
        return 
    
    def edge_based(self,graphs,output_path):
        examples = []
        for i, graph in enumerate(graphs):
            example = ""
            problemId = self.kb[self.dataset_name][self.problem_key].unique()[i]
            spl = problemId.split("_")
            prefix = problemId[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]
            
            example += self.truth_label(problemId,graph)

            node_mapping = {}
            # for j,node in enumerate(graph.x):
            #     if node[0] == 1:
            #         example += f"red(red).\n"
            #         node_mapping[j] = "red"
            #     elif node[1] == 1:
            #         example += f"green(green).\n"
            #         node_mapping[j] = "green"
                # elif node[2] == 1:
                #     example += f"blue(blue).\n"
                #     node_mapping[j] = "blue"
                # elif node[3] == 1:
                #     example += f"yellow(yellow).\n"
                #     node_mapping[j] = "yellow"
                # example += f"instance({prefix}{j}).\n"
            
            # add the unique node colors
            for color in current_objects["nodes"]["color"].unique():
                example += f"{color}({color}).\n"
            # add the edge between instances and colors
            for _,row in current_objects["nodes"].iterrows():
                example += f"instance({row['node_id']}).\n"
                example += f"edge({row['node_id']},{row['color']}).\n"

            for i,edge in enumerate(graph.edge_index.T):
                if edge[0] in node_mapping.keys():
                    node_1 = prefix + str(node_mapping[edge[0]])
                    node_2 = prefix + str(edge[1])
                    example += f"edge({node_1},{node_2}).\n"
                elif edge[1] in node_mapping.keys():
                    node_1 = prefix + str(edge[0])
                    node_2 = prefix + str(node_mapping[edge[1]])
                    example += f"edge({node_1},{node_2}).\n"
                else:
                    node_1 = prefix + str(edge[0].item())
                    node_2 = prefix + str(edge[1].item())
                    example += f"edge({node_1},{node_2}).\n"

            examples.append(example)
        
        with open(output_path, "w") as f:
            for example in examples:
                f.write(example)
        return
            

    def Klog(self,graphs,output_path):
        examples = []
        # add the colors
        colors = ""
        colors += "red(red).\n"
        colors += "green(green).\n"
        examples.append(colors)
        for i,graph in enumerate(graphs):
            example = ""
            problemId = self.kb[self.dataset_name][self.problem_key].unique()[i]
            spl = problemId.split("_")
            prefix = problemId[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]
            
            example += self.truth_label(problemId,graph)
            for j,node in enumerate(graph.x):
                if node[0] == 1:
                    node = prefix + str(j)
                    example += f"node({node},red).\n"
                elif node[1] == 1:
                    node = prefix + str(j)
                    example += f"node({node},green).\n"
                elif node[2] == 1:
                    node = prefix + str(j)
                    example += f"node({node},blue).\n"
                elif node[3] == 1:
                    node = prefix + str(j)
                    example += f"node({node},yellow).\n"
                elif node[-1] == 1:
                    node = prefix + str(j)
                    example += f"klog_edge({node}).\n"
            
            for i,edge in enumerate(graph.edge_index.T):
                node_1 = prefix + str(edge[0].item())
                node_2 = prefix + str(edge[1].item())
                example += f"edge({node_1},{node_2}).\n"
                    
            examples.append(example)
        
        with open(output_path, "w") as f:
            for example in examples:
                f.write(example)
        return


