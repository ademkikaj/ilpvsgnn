import torch
import pandas as pd
import os

class toLogic:

    def __init__(self,relational_path,dataset_name,movie_decoder,name_decoder,genre_decoder) -> None:
        self.relational_path = relational_path
        self.dataset_name = dataset_name
        self.movie_decoder = movie_decoder
        self.name_decoder = name_decoder
        self.genre_decoder = genre_decoder

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}

    def get_names(self,id):
        df = pd.read_csv(os.path.join(self.relational_path,f"{self.dataset_name}.csv"))
        name1 = df.iloc[id]['person1']
        name2 = df.iloc[id]['person2']
        return name1,name2
    

    def class_string(self,problem_id,graph):
        if graph.y.item() == 1:
            label = "pos"
        elif graph.y.item() == 0:
            label = "neg"
        return label
    
    
    def node_only(self,graphs,output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            label = self.class_string(problem_id,graph)
            name1,name2 = self.get_names(i)
            example += f"imdb({problem_id},{name1},{name2},{label}).\n"

            movie_start = len(self.name_decoder)
            movie_end = len(self.name_decoder) + len(self.movie_decoder)
            genre_start = movie_end
            genre_end = movie_end + len(self.genre_decoder)

            for j,node in enumerate(graph.x):
                name = self.name_decoder[tuple(node[0:len(self.name_decoder)].numpy())]
                example += f"person({problem_id},{nodeIds[j]},{name}).\n"
                if not(node[movie_start:movie_end] == 0.0).all().item():
                    movie = self.movie_decoder[tuple(node[movie_start:movie_end].numpy())]
                    example += f"movie({problem_id},{nodeIds[j]},{movie}).\n"
                if not(node[genre_start:genre_end] == 0.0).all().item():
                    genre = self.genre_decoder[tuple(node[genre_start:genre_end].numpy())]
                    example += f"genre({problem_id},{nodeIds[j]},{genre}).\n"
                if not(node[genre_end] == 0.0).all().item():
                    example += f"gender({problem_id},{nodeIds[j]},female).\n"
                if not(node[genre_end+1] == 0.0).all().item():
                    example += f"gender({problem_id},{nodeIds[j]},male).\n"
                if not(node[genre_end+2] == 0.0).all().item():
                    example += f"director({problem_id},{nodeIds[j]}).\n"
                if not(node[genre_end+3] == 0.0).all().item():
                    example += f"actor({problem_id},{nodeIds[j]}).\n"

            # add all the edges between all nodeIds
            for m in nodeIds:
                for n in nodeIds:
                    if m != n:
                        example += f"edge({problem_id},{nodeIds[m]},{nodeIds[n]}).\n"
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

            label = self.class_string(problem_id,graph)
            name1,name2 = self.get_names(i)
            example += f"imdb({problem_id},{name1},{name2},{label}).\n"

            movie_start = len(self.name_decoder)
            movie_end = len(self.name_decoder) + len(self.movie_decoder)

            for j,node in enumerate(graph.x):
                if not(node[0:len(self.name_decoder)] == 0.0).all().item():
                    name = self.name_decoder[tuple(node[0:len(self.name_decoder)].numpy())]
                    example += f"person({problem_id},{nodeIds[j]},{name}).\n"
                if not(node[movie_start:movie_end] == 0.0).all().item():
                    movie = self.movie_decoder[tuple(node[movie_start:movie_end].numpy())]
                    example += f"movie({problem_id},{nodeIds[j]},{movie}).\n"
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    edge_feat = graph.edge_attr[j]
                    if edge_feat[0] == 1:
                        relation = "in"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                    if edge_feat[1] == 1:
                        relation = "director"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                    if edge_feat[2] == 1:
                        relation = "actor"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"

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

            label = self.class_string(problem_id,graph)
            name1,name2 = self.get_names(i)
            example += f"imdb({problem_id},{name1},{name2},{label}).\n"

            movie_start = len(self.name_decoder)
            movie_end = len(self.name_decoder) + len(self.movie_decoder)

            for j,node in enumerate(graph.x):
                if not(node[0:len(self.name_decoder)] == 0.0).all().item():
                    name = self.name_decoder[tuple(node[0:len(self.name_decoder)].numpy())]
                    example += f"person({problem_id},{nodeIds[j]},{name}).\n"
                if not(node[movie_start:movie_end] == 0.0).all().item():
                    movie = self.movie_decoder[tuple(node[movie_start:movie_end].numpy())]
                    example += f"movie({problem_id},{nodeIds[j]},{movie}).\n"
                if not(node[-1] == 0.0).all().item():
                    example += f"instance({problem_id},{nodeIds[j]}).\n"
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    edge_feat = graph.edge_attr[j]
                    if edge_feat[0] == 1:
                        relation = "edge"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                    if edge_feat[1] == 1:
                        relation = "director"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                    if edge_feat[2] == 1:
                        relation = "actor"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
                    if edge_feat[3] == 1:
                        relation = "in"
                        example += f"{relation}({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"

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

            label = self.class_string(problem_id,graph)
            name1,name2 = self.get_names(i)
            example += f"imdb({problem_id},{name1},{name2},{label}).\n"

            movie_start = len(self.name_decoder)
            movie_end = len(self.name_decoder) + len(self.movie_decoder)

            for j,node in enumerate(graph.x):
                if not(node[0:len(self.name_decoder)] == 0.0).all().item():
                    name = self.name_decoder[tuple(node[0:len(self.name_decoder)].numpy())]
                    example += f"person({problem_id},{nodeIds[j]},{name}).\n"
                if not(node[movie_start:movie_end] == 0.0).all().item():
                    movie = self.movie_decoder[tuple(node[movie_start:movie_end].numpy())]
                    example += f"movie({problem_id},{nodeIds[j]},{movie}).\n"
                if not(node[-2] == 0.0).all().item():
                    example += f"director({problem_id},{nodeIds[j]}).\n"
                if not(node[-1] == 0.0).all().item():
                    example += f"actor({problem_id},{nodeIds[j]}).\n"
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    edge_feat = graph.edge_attr[j]
                    example += f"edge({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"

            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return
            