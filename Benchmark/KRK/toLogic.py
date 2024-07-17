import torch


class toLogic:

    def __init__(self,dataset_name,relational_path,problem_key) -> None:
        self.dataset_name = dataset_name

    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}
    
    
    def node_only(self,graphs,output_path):
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                if j == 0:
                    relation = "white_king"
                    example += f"{relation}({problem_id},{nodeIds[j]},{int(node[0].item())},{int(node[1].item())}).\n"
                elif j == 1:
                    relation = "white_rook"
                    example += f"{relation}({problem_id},{nodeIds[j]},{int(node[2].item())},{int(node[3].item())}).\n"
                elif j == 2:
                    relation = "black_king"
                    example += f"{relation}({problem_id},{nodeIds[j]},{int(node[4].item())},{int(node[5].item())}).\n"
            
            
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
    
            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            # add all atom relations
            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                if j == 0:
                    relation = "white_king"
                    example += f"{relation}({problem_id},{node_id},{int(node[0].item())},{int(node[1].item())}).\n"
                elif j == 1:
                    relation = "white_rook"
                    example += f"{relation}({problem_id},{node_id},{int(node[2].item())},{int(node[3].item())}).\n"
                elif j == 2:
                    relation = "black_king"
                    example += f"{relation}({problem_id},{node_id},{int(node[4].item())},{int(node[5].item())}).\n"
            
            # add all the bonds from the edge_index and edge features
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    edge_feat = graph.edge_attr[j]
                    relation = "edge"
                    example += f"{relation}({problem_id},{node_id1},{node_id2}).\n"
                    if edge_feat[0] == 1:
                        relation = "same_file"
                        example += f"{relation}({problem_id},{node_id1},{node_id2}).\n"
                    if edge_feat[1] == 1:
                        relation = "same_rank"
                        example += f"{relation}({problem_id},{node_id1},{node_id2}).\n"
            
            examples.append(example)
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
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                if j == 3:
                    relation = "white_king"
                    example += f"{relation}({problem_id},{nodeIds[j-3]},{int(node[3].item())},{int(node[4].item())}).\n"
                elif j == 4:
                    relation = "white_rook"
                    example += f"{relation}({problem_id},{nodeIds[j-3]},{int(node[5].item())},{int(node[6].item())}).\n"
                elif j == 5:
                    relation = "black_king"
                    example += f"{relation}({problem_id},{nodeIds[j-3]},{int(node[7].item())},{int(node[8].item())}).\n"
            
            
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

            # add the class example and
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            positions = [0,0,0,0,0,0]

            for j,node in enumerate(graph.x):

                if not (node[:2] == 0.0).all().item():
                    relation = "white_king"
                    example += f"{relation}({problem_id},{nodeIds[j]},{int(node[0].item())},{int(node[1].item())}).\n"
                    positions[0] = node[0].item()
                    positions[1] = node[1].item()
                if not (node[2:4] == 0.0).all().item():
                    relation = "white_rook"
                    example += f"{relation}({problem_id},{nodeIds[j]},{int(node[2].item())},{int(node[3].item())}).\n"
                    positions[2] = node[2].item()
                    positions[3] = node[3].item()
                if not (node[4:6] == 0.0).all().item():
                    relation = "black_king"
                    example += f"{relation}({problem_id},{nodeIds[j]},{int(node[4].item())},{int(node[5].item())}).\n"
                    positions[4] = node[4].item()
                    positions[5] = node[5].item()
            
            # add the same_file and same_rank nodes
            index = 3
            if positions[0] == positions[2]:
                example += f"same_file({problem_id},{nodeIds[index]}).\n"
                index += 1
            if positions[1] == positions[3]:
                example += f"same_rank({problem_id},{nodeIds[index]}).\n"
                index += 1
            if positions[0] == positions[4]:
                example += f"same_file({problem_id},{nodeIds[index]}).\n"
                index += 1
            if positions[1] == positions[5]:
                example += f"same_rank({problem_id},{nodeIds[index]}).\n"
                index += 1
            if positions[2] == positions[4]:
                example += f"same_file({problem_id},{nodeIds[index]}).\n"
                index += 1
            if positions[3] == positions[5]:
                example += f"same_rank({problem_id},{nodeIds[index]}).\n"
                index += 1
            
            # go over edges
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    example += f"edge({problem_id},{node_id1},{node_id2}).\n"
            
            examples.append(example)
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def FullBoard(self,graphs,output_path):
        examples = []
        nodeMapping = {
            bytes([0,0,0]) : "empty_cell",
            bytes([1,0,0]) : "white_king",
            bytes([0,1,0]) : "white_rook",
            bytes([0,0,1]) : "black_king",
            bytes([1,1,0]) : ["white_king","white_rook"],
            bytes([1,0,1]) : ["white_king","black_king"],
            bytes([0,1,1]) : ["white_rook","black_king"],
            bytes([1,1,1]) : ["white_king","white_rook","black_king"]
        }
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and
            if graph.y == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            # create the nodes
            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                relation = nodeMapping[bytes(node.int().tolist())]
                if isinstance(relation,list):
                    for r in relation:
                        example += f"{r}({problem_id},{node_id}).\n"
                else:
                    example += f"{relation}({problem_id},{node_id}).\n"
            # create the edges
            for j in graph.edge_index.T.tolist():
                node_id1 = nodeIds[j[0]]
                node_id2 = nodeIds[j[1]]
                example += f"edge({problem_id},{node_id1},{node_id2}).\n"
            examples.append(example)
        
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def FullDiag(self,graphs,output_path):
        self.FullBoard(graphs,output_path)
        return