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
                node_id = nodeIds[j]
                if not (node[:4] == 0.0).all().item():
                    relation = "drug"
                    example += f"{relation}({problem_id},{node_id},{node[0]},{node[1]},{node[2]},{node[3]}).\n"
                elif not (node[4] == 0.0).all().item():
                    relation = "atom"
                    example += f"{relation}({problem_id},{node_id},{node[4]}).\n"
                elif not (node[5] == 0.0).all().item():
                    relation = "bond"
                    example += f"{relation}({problem_id},{node_id},{node[5]}).\n"
                elif not torch.all(node[6]):
                    relation = "nitro"
                    example += f"{relation}({problem_id},{node_id}).\n"
                elif not torch.all(node[7]):
                    relation = "benzene"
                    example += f"{relation}({problem_id},{node_id}).\n"  
            
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
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            # add all atom relations
            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                relation = "atom"
                example += f"{relation}({problem_id},{node_id},{int(node[0].item())},{int(node[1].item())}).\n"
            
            # add all the bonds from the edge_index and edge features
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    edge_feat = graph.edge_attr[j]
                    relation = "bond"
                    example += f"{relation}({problem_id},{node_id1},{node_id2},{edge_feat[0]}).\n"
                    if edge_feat[1] == 1:
                        relation = "nitro"
                        example += f"{relation}({problem_id},{node_id2},{node_id1}).\n"
                    if edge_feat[2] == 1:
                        relation = "benzene"
                        example += f"{relation}({problem_id},{node_id2},{node_id1}).\n"
            
            
            examples.append(example)
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def edge_based(self, graphs, output_path):
        examples = []

        # adding the 3 shapes as facts to the file
        #examples.append("square.\ncircle.\ntriangle.\n")

        shape_mapping = {0:"atom"}

        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0]-1)

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"{self.dataset_name}({problem_id},{label}).\n"

            node_mapping = {bytes([0,0,0,1]): "shape"}

            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    # if the edge is external (between shape instance and shape fact, than also the shape instance is added)
                    if edge[0] == 0:
                        shape = shape_mapping[edge[0]]
                        node_id = nodeIds[edge[1]-1]
                        example += f"instance({problem_id},{shape},{node_id}).\n"
                    elif edge[1] == 0:
                        shape = shape_mapping[edge[1]]
                        node_id = nodeIds[edge[0]-1]
                        example += f"instance({problem_id},{shape},{node_id}).\n"
                    else:
                        edge_feat = graph.edge_attr[j]
                        node_id1 = nodeIds[edge[0]-1]
                        node_id2 = nodeIds[edge[1]-1]
                        example += f"bond({problem_id},{node_id1},{node_id2},{edge_feat[2]}).\n"
                        if edge_feat[3] == 1:
                            relation = "nitro"
                            example += f"{relation}({problem_id},{node_id2},{node_id1}).\n"
                        if edge_feat[4] == 1:
                            relation = "benzene"
                            example += f"{relation}({problem_id},{node_id2},{node_id1}).\n"
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def Klog(self,graphs,output_path):
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

            # go over the nodes
            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                if not (node[:2] == 0.0).all().item():
                    relation = "atom"
                    example += f"{relation}({problem_id},{node_id},{int(node[0].item())},{int(node[1].item())}).\n"
                elif not (node[2] == 0.0):
                    relation = "bond"
                    example += f"{relation}({problem_id},{node_id}).\n"
                elif not (node[3] == 0.0):
                    relation = "nitro"
                    example += f"{relation}({problem_id},{node_id}).\n"
                elif not (node[4] == 0.0):
                    relation = "benzene"
                    example += f"{relation}({problem_id},{node_id}).\n"
            
            # go over the relations
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    edge_feat = graph.edge_attr[j]
                    relation = "edge"
                    example += f"{relation}({problem_id},{node_id1},{node_id2}).\n"

            examples.append(example)
        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def VirtualNode(self,graphs,output_path):
        nodeMapping = {
            bytes([1,0,0,0,0,0,0,0]) : 'c',
            bytes([0,1,0,0,0,0,0,0]) : 'n',
            bytes([0,0,1,0,0,0,0,0]) : 'o',
            bytes([0,0,0,1,0,0,0,0]) : 'h',
            bytes([0,0,0,0,1,0,0,0]) : 'cl',
            bytes([0,0,0,0,0,1,0,0]) : 'f',
            bytes([0,0,0,0,0,0,1,0]) : 'br',
            bytes([0,0,0,0,0,0,0,1]) : 'i',
            bytes([0,0,0,0,0,0,0,0]) : 'drug'
        }
        edgeMapping = {
            bytes([1,0,0,0,0,0]) : "first",
            bytes([0,1,0,0,0,0]) : "second",
            bytes([0,0,1,0,0,0]) : "third",
            bytes([0,0,0,1,0,0]) : "fourth",
            bytes([0,0,0,0,1,0]) : "fifth",
            bytes([0,0,0,0,0,1]) : "seventh"
        }
        examples = []
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0]-1)

            # add the class example with the problem id and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"mutag({problem_id},{label}).\n"

            # create the nodes
            for j,node in enumerate(graph.x):
                first_node = node[:8]
                relation = nodeMapping[bytes(first_node.int().tolist())]
                if relation == 'drug':
                    ind1 = node[8].item()
                    inda = node[9].item()
                    logp = node[10].item()
                    lumo = node[11].item()
                    example += f"{relation}({problem_id},{ind1},{inda},{logp},{lumo}).\n"
                else:
                    node_id = nodeIds[j]
                    example += f"{relation}({problem_id},{node_id}).\n"
            # create the edges
            drug_node_index = graph.x.shape[0]-1
            edges = []
            for j in graph.edge_index.T.tolist():
                if j not in edges and drug_node_index not in j:
                    edges.append(j)
            for j,edge in enumerate(edges):
                node_id1 = nodeIds[edge[0]]
                node_id2 = nodeIds[edge[1]]
                edge_attr = edgeMapping[bytes(graph.edge_attr[j].int().tolist())]
                example += f"{edge_attr}({problem_id},{node_id1},{node_id2}).\n"
            
            examples.append(example)

        with open(output_path,'w') as f:
            for ex in examples:
                f.write(ex)

                