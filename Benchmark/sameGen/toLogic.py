import torch
import pandas as pd
import os

class toLogic:

    def __init__(self,dataset_name,name_decoder,relational_path) -> None:
        self.dataset_name = dataset_name
        self.name_decoder = name_decoder
        self.relational_path = relational_path

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}
    

    def class_string(self,problem_id,graph):
        if graph.y.item() == 1:
            label = "pos"
        elif graph.y.item() == 0:
            label = "neg"
        return label

    def get_names(self,id):
        df = pd.read_csv(os.path.join(self.relational_path,f"{self.dataset_name}.csv"))
        name1 = df.iloc[id]['name1']
        name2 = df.iloc[id]['name2']
        return name1,name2
    
    def node_only(self,graphs,output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            label = self.class_string(problem_id,graph)
            name1,name2 = self.get_names(i)
            example += f"sameGen({problem_id},{name1},{name2},{label}).\n"

            for j,node in enumerate(graph.x):
                if not (node[0].item() == 0.0):
                    relation = "parent"
                    name = self.name_decoder[tuple(node[2:].numpy())]
                    example += f"{relation}({problem_id},{nodeIds[j]},{name}).\n"
                elif not(node[1].item() == 0.0):
                    relation = "child"
                    name = self.name_decoder[tuple(node[2:].numpy())]
                    example += f"{relation}({problem_id},{nodeIds[j]},{name}).\n"
                else:
                    relation = "person"
                    name = self.name_decoder[tuple(node[2:].numpy())]
                    example += f"{relation}({problem_id},{nodeIds[j]},{name}).\n"
                examples.append(example)

            # add all the edges between all nodeIds
            for m in nodeIds:
                for n in nodeIds:
                    if m != n:
                        example += f"edge({problem_id},{m},{n}).\n"
            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def node_edge(self,graphs,output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            name1,name2 = self.get_names(i)
            label = self.class_string(problem_id,graph)
            example += f"sameGen({problem_id},{name1},{name2},{label}).\n"

            node_mapping = {}
            for j,node in enumerate(graph.x):
                relation = "person"
                name = self.name_decoder[tuple(node.numpy())]
                example += f"{relation}({problem_id},{name}).\n"
                node_mapping[j] = name
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    edge_feat = graph.edge_attr[j]
                    if edge_feat[0].item() == 1:
                        relation = "parent"
                        parent_name = node_mapping[edge[0]]
                        child_name = node_mapping[edge[1]]
                        example += f"{relation}({problem_id},{parent_name},{child_name}).\n"
                    if edge_feat[1].item() == 1:
                        relation = "same_gen"
                        name1 = node_mapping[edge[0]]
                        name2 = node_mapping[edge[1]]
                        example += f"{relation}({problem_id},{name1},{name2}).\n"

            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def edge_based(self, graphs, output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            name1,name2 = self.get_names(i)
            label = self.class_string(problem_id,graph)
            example += f"sameGen({problem_id},{name1},{name2},{label}).\n"

            node_mapping = {}
            for j,node in enumerate(graph.x):
                if node[0].item() == 1.0:
                    relation = "instance"
                    example += f"{relation}({problem_id},{nodeIds[j]}).\n"
                    node_mapping[j] = nodeIds[j]
                else:
                    relation = "person"
                    name = self.name_decoder[tuple(node[1:].numpy())]
                    example += f"{relation}({problem_id},{nodeIds[j]},{name}).\n"
                    node_mapping[j] = nodeIds[j]
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    edge_feat = graph.edge_attr[j]
                    if edge_feat[0].item() == 1:
                        relation = "edge"
                        example += f"{relation}({problem_id},{node_mapping[edge[0]]},{node_mapping[edge[1]]}).\n"
                    if edge_feat[2].item() == 1:
                        relation = "parent"
                        example += f"{relation}({problem_id},{node_mapping[edge[0]]},{node_mapping[edge[1]]}).\n"
                    if edge_feat[3].item() == 1:
                        relation = "same_gen"
                        example += f"{relation}({problem_id},{node_mapping[edge[0]]},{node_mapping[edge[1]]}).\n"

            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def Klog(self,graphs,file):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            name1,name2 = self.get_names(i)
            label = self.class_string(problem_id,graph)
            example += f"sameGen({problem_id},{name1},{name2},{label}).\n"

            node_mapping = {}
            name_mapping = {}
            for j,node in enumerate(graph.x):
                if (node[:2] == 0.0).all():
                    relation = "person"
                    name = self.name_decoder[tuple(node[2:].numpy())]
                    example += f"{relation}({problem_id},{nodeIds[j]},{name}).\n"
                    name_mapping[j] = name
                    node_mapping[j] = nodeIds[j]
                else:
                    if (node[0].item() == 1.0):
                        relation = "parent"
                        example += f"{relation}({problem_id},{nodeIds[j]}).\n"
                        name_mapping[j] = nodeIds[j]
                        node_mapping[j] = nodeIds[j]
                    elif (node[1].item() == 1.0):
                        relation = "same_gen"
                        example += f"{relation}({problem_id},{nodeIds[j]}).\n"
                        name_mapping[j] = nodeIds[j]
                        node_mapping[j] = nodeIds[j]
            
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    relation = "edge"
                    example += f"{relation}({problem_id},{node_mapping[edge[0]]},{node_mapping[edge[1]]}).\n"

            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return
            