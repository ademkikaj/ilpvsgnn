from dataclasses import dataclass
import torch
import os

class toLogic:

    def __init__(self,dataset_name,relational_path,problem_key) -> None:
        self.dataset_name = dataset_name
    
    def createNodeIds(self,n):
        return {i:'n'+str(i) for i in range(n)}

    def node_only(self, graphs, output_path):
        #graphs = torch.load(os.path.join(self.exp_path,'graph/node_only.pt'))
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
            example += f"bongard({problem_id},{label}).\n"

            node_mapping = {
                bytes([1,0,0,0]): "square",
                bytes([0,1,0,0]): "circle",
                bytes([0,0,1,0]): "triangle",
                bytes([0,0,0,1]): "in"
            }

            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                relation = node_mapping[bytes(node.int().tolist())]
                example += f"{relation}({problem_id},{node_id}).\n"
            
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

    def node_edge(self,graphs,file):
        examples = []
        print("Graph length: ",len(graphs))
        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])
        
        
            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"bongard({problem_id},{label}).\n"

            shapes = ["triangle", "square", "circle","shape1","shape2","shape3","shape4","shape5"]
            num_node_features = graph.x.shape[1]
            node_mapping = {}
            for i,shape in enumerate(shapes):
                if i < num_node_features:
                    node_mapping[shape] = [0]*num_node_features
                    node_mapping[shape][i] = 1
                    node_mapping[shape] = bytes(node_mapping[shape])
            node_mapping = {v:k for k,v in node_mapping.items()}
            # node_mapping = {
            #     bytes([1,0,0]): "square",
            #     bytes([0,1,0]): "circle",
            #     bytes([0,0,1]): "triangle"
            # }

            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                relation = node_mapping[bytes(node.int().tolist())]
                example += f"{relation}({problem_id},{node_id}).\n"
            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    example += f"edge({problem_id},{node_id1},{node_id2}).\n"
            
            
            examples.append(example)

        print("Average nodes: ",sum([len(graph.x) for graph in graphs])/len(graphs))
        print("average edges: ",sum([len(graph.edge_index.T) for graph in graphs])/len(graphs))
        #file = self.exp_path + "/logic/node_edge.pl"
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return
    
    def edge_based(self,graphs,file):
        examples = []

        # adding the 3 shapes as facts to the file
        examples.append("square(square).\ncircle(circle).\ntriangle(triangle).\n")

        shape_mapping = {0:"square",1:"circle",2:"triangle"}

        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0]-3)

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"bongard({problem_id},{label}).\n"

            node_mapping = {bytes([0,0,0,1]): "shape"}

            for j,node in enumerate(graph.x):
                if j>=3:
                    node_id = nodeIds[j-3]
                    relation = "instance"
                    example += f"{relation}({problem_id},{node_id}).\n"

            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j % 2 == 0:
                    # if the edge is external (between shape instance and shape fact, than also the shape instance is added)
                    if edge[0] < 3:
                        shape = shape_mapping[edge[0]]
                        node_id = nodeIds[edge[1]-3]
                        example += f"shape({problem_id},{shape},{node_id}).\n"
                    elif edge[1] < 3:
                        shape = shape_mapping[edge[1]]
                        node_id = nodeIds[edge[0]-3]
                        example += f"shape({problem_id},{shape},{node_id}).\n"
                    else:
                        # just add the in relation
                        node_id1 = nodeIds[edge[0]-3]
                        node_id2 = nodeIds[edge[1]-3]
                        example += f"in({problem_id},{node_id1},{node_id2}).\n"
            
            
            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return

    def Klog(self,graphs,file):
        examples = []
        node_mapping = {
            bytes([1,0,0,0]): "square",
            bytes([0,1,0,0]): "circle",
            bytes([0,0,1,0]): "triangle",
            bytes([0,0,0,1]): "in"
        }

        for i,graph in enumerate(graphs):
            example = ""
            problem_id = str(i)
            nodeIds = self.createNodeIds(graph.x.shape[0])

            # add the class example and the truth label
            if graph.y.item() == 1:
                label = "pos"
            else:
                label = "neg"
            example += f"bongard({problem_id},{label}).\n"

            for j,node in enumerate(graph.x):
                node_id = nodeIds[j]
                relation = node_mapping[bytes(node.int().tolist())]
                example += f"{relation}({problem_id},{node_id}).\n"
            
            # the edge indices are in relations between nodeIds
            for j,edge in enumerate(graph.edge_index.T.tolist()):
                if j%2 == 0:
                    node_id1 = nodeIds[edge[0]]
                    node_id2 = nodeIds[edge[1]]
                    example += f"edge({problem_id},{node_id1},{node_id2}).\n"

            examples.append(example)

        #file = self.exp_path + "/logic/edge_based.pl"
        with open(file,'w') as f:
            for ex in examples:
                f.write(ex)
        return




# def createNodeIds(n):
#     return {i:'n'+str(i) for i in range(n)}


# ### HotIn ### 
# """ No edge features and no node features (in the ILP syntax)"""

# dataset = BongardDataset(root='datasets/Bongard/HotIn')
# datalist = dataset.data_list

# nodeMapping = {
#     bytes([0,0,0,0]): "frame",
#     bytes([1,0,0,0]): "square",
#     bytes([0,1,0,0]): "circle",
#     bytes([0,0,1,0]): "triangleUp",
#     bytes([0,0,0,1]): "triangleDown"
# }


# examples = []
# examples_aleph = []
# bg_aleph = []
# for i,graph in enumerate(datalist):
#     example_ILP = ""
#     problem_id = str(i)
#     nodeIds = createNodeIds(graph.x.shape[0])

#     # add the class example with the problem id and the truth label
#     if graph.y.item() == 1:
#         label = "pos"
#     else:
#         label = "neg"
#     example_ILP += f"bongard({problem_id},{label}).\n"
#     examples_aleph.append(f"{label}(bongard({problem_id})).\n")
    
#     # create the nodes
#     for j,node in enumerate(graph.x):
#         node_id = nodeIds[j]
#         relation = nodeMapping[bytes(node.int().tolist())]
#         example_ILP += f"{relation}({problem_id},{node_id}).\n"
#         bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
#     # create the edges
    
#     # extract unique edges
#     edges = []
#     for j in graph.edge_index.T.tolist():
#         if j not in edges:
#             edges.append(j)
#     for j,edge in enumerate(edges):
#         node_id1 = nodeIds[edge[0]]
#         node_id2 = nodeIds[edge[1]]
#         example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
#         bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
#     examples.append(example_ILP)
    
# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/HotIn/bongard.kb"
# with open(file_path,'w') as f:
#     for example in examples:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/HotIn/exs.pl"
# with open(file_path,'w') as f:
#     for example in examples_aleph:
#         f.write(example)
# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/HotIn/bk.pl"
# with open(file_path,'w') as f:
#     for example in bg_aleph:
#         f.write(example)


# ### HotKlog ###
# """ No edge features and node features (in the ILP syntax)"""
# dataset = BongardDataset(root='datasets/Bongard/HotKlog')
# datalist = dataset.data_list

# nodeMapping = {
#     bytes([0,0,0,0,0]): "frame",
#     bytes([1,0,0,0,0]): "square",
#     bytes([0,1,0,0,0]): "circle",
#     bytes([0,0,1,0,0]): "triangleUp",
#     bytes([0,0,0,1,0]): "triangleDown",
#     bytes([0,0,0,0,1]): "in"
# }

# examples = []
# examples_aleph = []
# bg_aleph = []
# for i,graph in enumerate(datalist):
#     example_ILP = ""
#     problem_id = str(i)
#     nodeIds = createNodeIds(graph.x.shape[0])

#     # add the class example with the problem id and the truth label
#     if graph.y.item() == 1:
#         label = "pos"
#     else:
#         label = "neg"
#     example_ILP += f"bongard({problem_id},{label}).\n"
#     examples_aleph.append(f"{label}(bongard({problem_id})).\n")
    
#     # create the nodes
#     for j,node in enumerate(graph.x):
#         node_id = nodeIds[j]
#         relation = nodeMapping[bytes(node.int().tolist())]
#         example_ILP += f"{relation}({problem_id},{node_id}).\n"
#         bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
#     # create the edges
    
#     # extract unique edges
#     edges = []
#     for j in graph.edge_index.T.tolist():
#         if j not in edges and j.reverse() not in edges:
#             edges.append(j)
#     for j,edge in enumerate(edges):
#         node_id1 = nodeIds[edge[0]]
#         node_id2 = nodeIds[edge[1]]
#         example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
#         bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
#     examples.append(example_ILP)
    

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/HotKlog/bongard.kb"
# with open(file_path,'w') as f:
#     for example in examples:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/HotKlog/exs.pl"
# with open(file_path,'w') as f:
#     for example in examples_aleph:
#         f.write(example)
# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/HotKlog/bk.pl"
# with open(file_path,'w') as f:
#     for example in bg_aleph:
#         f.write(example)

# ### HotEdgesFull ###
# """ 
# Edge features and no node features (in the ILP syntax)
# The different edge features are mapped to different edge relations.
# """


# dataset = BongardDataset(root='datasets/Bongard/HotEdgesFull')
# datalist = dataset.data_list

# nodeMapping = {
#     bytes([0,0,0,0]): "frame",
#     bytes([1,0,0,0]): "square",
#     bytes([0,1,0,0]): "circle",
#     bytes([0,0,1,0]): "triangleUp",
#     bytes([0,0,0,1]): "triangleDown"
# }

# edgeMapping = {
#     0 : "edge",
#     1 : "in"
# }

# examples = []
# examples_aleph = []
# bg_aleph = []
# for i,graph in enumerate(datalist):
#     example_ILP = ""
#     problem_id = str(i)
#     nodeIds = createNodeIds(graph.x.shape[0])

#     # add the class example with the problem id and the truth label
#     if graph.y.item() == 1:
#         label = "pos"
#     else:
#         label = "neg"
#     example_ILP += f"bongard({problem_id},{label}).\n"
#     examples_aleph.append(f"{label}(bongard({problem_id})).\n")
    
#     # create the nodes
#     for j,node in enumerate(graph.x):
#         node_id = nodeIds[j]
#         relation = nodeMapping[bytes(node.int().tolist())]
#         example_ILP += f"{relation}({problem_id},{node_id}).\n"
#         bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
#     # create the edges
#     # extract unique edges
#     edges = []
#     attributes = []
#     for j,edge in enumerate(graph.edge_index.T.tolist()):
#         if edge not in edges and edge.reverse() not in edges:
#             edges.append(edge)
#             attributes.append(graph.edge_attr[j].item())
#     for j,edge in enumerate(edges):
#         edge_relation = edgeMapping[attributes[j]]
#         node_id1 = nodeIds[edge[0]]
#         node_id2 = nodeIds[edge[1]]
#         example_ILP += f"{edge_relation}({problem_id},{node_id1},{node_id2}).\n"
#         bg_aleph.append(f"{edge_relation}({problem_id},{node_id1},{node_id2}).\n")
    
#     examples.append(example_ILP)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/HotEdges/bongard.kb"

# with open(file_path,'w') as f:
#     for example in examples:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/HotEdges/exs.pl"
# with open(file_path,'w') as f:
#     for example in examples_aleph:
#         f.write(example)
# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/HotEdges/bk.pl"
# with open(file_path,'w') as f:
#     for example in bg_aleph:
#         f.write(example)



# ### NoFrame ###

# dataset = BongardDataset(root='datasets/Bongard/NoFrame')
# datalist = dataset.data_list

# nodeMapping = {
#     bytes([1,0,0,0]): "square",
#     bytes([0,1,0,0]): "circle",
#     bytes([0,0,1,0]): "triangleUp",
#     bytes([0,0,0,1]): "triangleDown"
# }

# examples = []
# examples_aleph = []
# bg_aleph = []
# for i,graph in enumerate(datalist):
#     example_ILP = ""
#     problem_id = str(i)
#     nodeIds = createNodeIds(graph.x.shape[0])

#     # add the class example with the problem id and the truth label
#     if graph.y.item() == 1:
#         label = "pos"
#     else:
#         label = "neg"
#     example_ILP += f"bongard({problem_id},{label}).\n"
#     examples_aleph.append(f"{label}(bongard({problem_id})).\n")
    
#     # create the nodes
#     for j,node in enumerate(graph.x):
#         node_id = nodeIds[j]
#         relation = nodeMapping[bytes(node.int().tolist())]
#         example_ILP += f"{relation}({problem_id},{node_id}).\n"
#         bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
#     # create the edges
    
#     # extract unique edges
#     edges = []
#     for j in graph.edge_index.T.tolist():
#         if j not in edges:
#             edges.append(j)
#     for j,edge in enumerate(edges):
#         node_id1 = nodeIds[edge[0]]
#         node_id2 = nodeIds[edge[1]]
#         example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
#         bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
#     examples.append(example_ILP)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/NoFrame/bongard.kb"
# with open(file_path,'w') as f:
#     for example in examples:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/NoFrame/exs.pl"
# with open(file_path,'w') as f:
#     for example in examples_aleph:
#         f.write(example)
# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/NoFrame/bk.pl"
# with open(file_path,'w') as f:
#     for example in bg_aleph:
#         f.write(example)


# ### NodesOnly ###

# dataset = BongardDataset(root='datasets/Bongard/NodesOnly')
# datalist = dataset.data_list

# nodeMapping = {
#     bytes([0,0,0,0]): "frame",
#     bytes([1,0,0,0]): "square",
#     bytes([0,1,0,0]): "circle",
#     bytes([0,0,1,0]): "triangleUp",
#     bytes([0,0,0,1]): "triangleDown"
# }

# examples = []
# examples_aleph = []
# bg_aleph = []
# for i,graph in enumerate(datalist):
#     example_ILP = ""
#     problem_id = str(i)
#     nodeIds = createNodeIds(graph.x.shape[0])

#     # add the class example with the problem id and the truth label
#     if graph.y.item() == 1:
#         label = "pos"
#     else:
#         label = "neg"
#     example_ILP += f"bongard({problem_id},{label}).\n"
#     examples_aleph.append(f"{label}(bongard({problem_id})).\n")
    
#     # create the nodes
#     for j,node in enumerate(graph.x):
#         node_id = nodeIds[j]
#         relation = nodeMapping[bytes(node.int().tolist())]
#         example_ILP += f"{relation}({problem_id},{node_id}).\n"
#         bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
#     # create the edges
    
#     # extract unique edges
#     edges = []
#     for j in graph.edge_index.T.tolist():
#         if j not in edges and j.reverse() not in edges:
#             edges.append(j)
#     for j,edge in enumerate(edges):
#         node_id1 = nodeIds[edge[0]]
#         node_id2 = nodeIds[edge[1]]
#         example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
#         bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
#     examples.append(example_ILP)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/NodesOnly/bongard.kb"
# with open(file_path,'w') as f:
#     for example in examples:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/NodesOnly/exs.pl"
# with open(file_path,'w') as f:
#     for example in examples_aleph:
#         f.write(example)
# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/NodesOnly/bk.pl"

# with open(file_path,'w') as f:
#     for example in bg_aleph:
#         f.write(example)




# ### CorrectKlog ###
        
# dataset = BongardDataset(root='datasets/Bongard/CorrectKlog')
# datalist = dataset.data_list
        
# nodeMapping = {
#     bytes([0,0,0,0,0,0]) : "frame",
#     bytes([1,0,0,0,0,0]) : "square",
#     bytes([0,1,0,0,0,0]) : "circle",
#     bytes([0,0,1,0,0,0]) : "triangleUp",
#     bytes([0,0,0,1,0,0]) : "triangleDown",
#     bytes([0,0,0,0,1,0]) : "in",
#     bytes([0,0,0,0,0,1]) : "partOf"
# }

# examples = []
# examples_aleph = []
# bg_aleph = []
# for i,graph in enumerate(datalist):
#     example_ILP = ""
#     problem_id = str(i)
#     nodeIds = createNodeIds(graph.x.shape[0])

#     # add the class example with the problem id and the truth label
#     if graph.y.item() == 1:
#         label = "pos"
#     else:
#         label = "neg"
#     example_ILP += f"bongard({problem_id},{label}).\n"
#     examples_aleph.append(f"{label}(bongard({problem_id})).\n")
    
#     # create the nodes
#     for j,node in enumerate(graph.x):
#         node_id = nodeIds[j]
#         relation = nodeMapping[bytes(node.int().tolist())]
#         example_ILP += f"{relation}({problem_id},{node_id}).\n"
#         bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
#     # create the edges
    
#     # extract unique edges
#     edges = []
#     for j in graph.edge_index.T.tolist():
#         if j not in edges and j.reverse() not in edges:
#             edges.append(j)
#     for j,edge in enumerate(edges):
#         node_id1 = nodeIds[edge[0]]
#         node_id2 = nodeIds[edge[1]]
#         example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
#         bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
#     examples.append(example_ILP)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/CorrectKlog/bongard.kb"
# with open(file_path,'w') as f:
#     for example in examples:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/CorrectKlog/exs.pl"
# with open(file_path,'w') as f:
#     for example in examples_aleph:
#         f.write(example)

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Aleph/CorrectKlog/bk.pl"
# with open(file_path,'w') as f:
#     for example in bg_aleph:
#         f.write(example)
