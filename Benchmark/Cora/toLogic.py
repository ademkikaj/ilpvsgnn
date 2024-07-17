import torch


class toLogic:

    def __init__(self,dataset_name) -> None:
        self.dataset_name = dataset_name

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}
    

    def class_string(self,problem_id,graph):
        if graph.y.item() == 1:
            label = "pos"
        elif graph.y.item() == 0:
            label = "neg"
        return f"{self.dataset_name}({problem_id},{label}).\n"
    
    
    def node_only(self,graphs,output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            
            example += self.class_string(problem_id,graph)

            for j,node in enumerate(graph.x):
                if not (node[0].item() == 0.0):
                    relation = "students"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[0].item()}).\n"
                    student_id = nodeIds[j]

                if not(node[1:3] == 0.0).all().item():
                    relation = "course"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[1].item()},{node[2].item()}).\n"
                if not(node[3:5] == 0.0).all().item():
                    relation = "professor"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[3].item()},{node[4].item()}).\n"
                if not(node[5:7] == 0.0).all().item():
                    relation = "registered"
                    example += f"{relation}({problem_id},{nodeIds[node[5].item()]},{node[6].item()},{node[7].item()}).\n"
                if not(node[7:9] == 0.0).all().item():
                    relation = "ra"
                    example += f"{relation}({problem_id},{nodeIds[node[8].item()]},{node[9].item()},{node[10].item()}).\n"
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
            example += self.class_string(problem_id,graph)

            for j,node in enumerate(graph.x):
                if not (node[0].item() == 0.0):
                    relation = "students"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[0].item()}).\n"

                if not(node[1:3] == 0.0).all().item():
                    relation = "course"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[1].item()},{node[2].item()}).\n"
                if not(node[3:5] == 0.0).all().item():
                    relation = "professor"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[3].item()},{node[4].item()}).\n"
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    edge_feat = graph.edge_attr[j]
                    if not(edge_feat[0:2] == 0.0).all().item():
                        relation = "registered"
                        student_id = nodeIds[edge[0]]
                        course_id = nodeIds[edge[1]]
                        example += f"{relation}({problem_id},{student_id},{course_id},{edge_feat[0].item()},{edge_feat[1].item()}).\n"
                    if not(edge_feat[2:4] == 0.0).all().item():
                        relation = "ra"
                        professor_id = nodeIds[edge[0]]
                        course_id = nodeIds[edge[1]]
                        example += f"{relation}({problem_id},{professor_id},{course_id},{edge_feat[2].item()},{edge_feat[3].item()}).\n"

            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def edge_based(self, graphs, output_path):
        examples = []

        # adding the 3 shapes as facts to the file
        #examples.append("square.\ncircle.\ntriangle.\n")

        piece_mapping = {0:"white_king",1:"white_rook",2:"black_king"}

        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0]-len(piece_mapping))

            # add the class example and the truth label
            example += self.class_string(problem_id,graph)

            for j,node in enumerate(graph.x):
                if j == 3:
                    relation = "white_king"
                    example += f"{relation}({problem_id},{nodeIds[j-3]},{node[3].item()},{node[4].item()}).\n"
                elif j == 4:
                    relation = "white_rook"
                    example += f"{relation}({problem_id},{nodeIds[j-3]},{node[5].item()},{node[6].item()}).\n"
                elif j == 5:
                    relation = "black_king"
                    example += f"{relation}({problem_id},{nodeIds[j-3]},{node[7].item()},{node[8].item()}).\n"
            
            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    # if the edge is external (between shape instance and shape fact, than also the shape instance is added)
                    if edge[0] < 3:
                        shape = piece_mapping[edge[0]]
                        node_id = nodeIds[edge[1]-3]
                        example += f"instance({problem_id},{shape},{node_id}).\n"
                    elif edge[1] < 3:
                        shape = piece_mapping[edge[1]]
                        node_id = nodeIds[edge[0]-3]
                        example += f"instance({problem_id},{shape},{node_id}).\n"
                    else:
                        edge_feat = graph.edge_attr[j]
                        node_id1 = nodeIds[edge[0]-3]
                        node_id2 = nodeIds[edge[1]-3]
                        example += f"edge({problem_id},{node_id1},{node_id2}).\n"
                        if edge_feat[2] == 1:
                            relation = "same_file"
                            example += f"{relation}({problem_id},{node_id2},{node_id1}).\n"
                        if edge_feat[3] == 1:
                            relation = "same_rank"
                            example += f"{relation}({problem_id},{node_id2},{node_id1}).\n"
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
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
            example += self.class_string(problem_id,graph)

            for j,node in enumerate(graph.x):
                if not (node[0].item() == 0.0):
                    relation = "students"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[0].item()}).\n"

                if not(node[1:3] == 0.0).all().item():
                    relation = "course"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[1].item()},{node[2].item()}).\n"
                if not(node[3:5] == 0.0).all().item():
                    relation = "professor"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[3].item()},{node[4].item()}).\n"
                if not(node[5:7] == 0.0).all().item():
                    relation = "registered"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[5].item()},{node[6].item()}).\n"
                if not(node[7:9] == 0.0).all().item():
                    relation = "ra"
                    example += f"{relation}({problem_id},{nodeIds[j]},{node[7].item()},{node[8].item()}).\n"
            
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    example += f"edge({problem_id},{nodeIds[edge[0]]},{nodeIds[edge[1]]}).\n"
    
            examples.append(example)
        #file = self.exp_path + "/logic/node_only.pl"
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return
            