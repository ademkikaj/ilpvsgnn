from KRKDataset import KRKDataset
import torch


def createNodeIds(n):
    return {i:'n'+str(i) for i in range(n)}

### FullBoard ###
""" 
No edge features and no node features (in the ILP syntax)
"""

dataset = KRKDataset(root='datasets/KRK/FullBoard')
datalist = dataset.data_list

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

examples = []
examples_aleph = []
bg_aleph = []
for i, graph in enumerate(datalist):
    example_ILP = ""
    problem_id = str(i)
    nodeIds = createNodeIds(graph.x.shape[0])

    # add the class example with the problem id and the truth label
    label = "pos" if graph.y.item() == 1 else "neg"
    example_ILP += f"krk({problem_id},{label}).\n"
    examples_aleph.append(f"{label}(krk({problem_id})).\n")

    # create the nodes
    for j, node in enumerate(graph.x):
        node_id = nodeIds[j]
        relation = nodeMapping[bytes(node.int().tolist())]
        if isinstance(relation,list):
            for r in relation:
                example_ILP += f"{r}({problem_id},{node_id}).\n"
                bg_aleph.append(f"{r}({problem_id},{node_id}).\n")
        else:
            example_ILP += f"{relation}({problem_id},{node_id}).\n"
            bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")

    # create the edges
    edges = []
    for j in graph.edge_index.T.tolist():
        if j not in edges:
            edges.append(j)
    for j, edge in enumerate(edges):
        node_id1 = nodeIds[edge[0]]
        node_id2 = nodeIds[edge[1]]
        example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
        bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
    examples.append(example_ILP)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/KRK/FullBoard/krk.kb"
with open(file_path,'w') as f:
    for example in examples:
        f.write(example)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Aleph/FullBoard/exs.pl"
with open(file_path,'w') as f:
    for example in examples_aleph:
        f.write(example)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Aleph/FullBoard/bk.pl"
with open(file_path,'w') as f:
    for example in bg_aleph:
        f.write(example)





### FullDiag ###
""" 
No edge features and no node features (in the ILP syntax)
"""

dataset = KRKDataset(root='datasets/KRK/FullDiag')
datalist = dataset.data_list

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

examples = []
examples_aleph = []
bg_aleph = []
for i, graph in enumerate(datalist):
    example_ILP = ""
    problem_id = str(i)
    nodeIds = createNodeIds(graph.x.shape[0])

    # add the class example with the problem id and the truth label
    label = "pos" if graph.y.item() == 1 else "neg"
    example_ILP += f"krk({problem_id},{label}).\n"
    examples_aleph.append(f"{label}(krk({problem_id})).\n")

    # create the nodes
    for j, node in enumerate(graph.x):
        node_id = nodeIds[j]
        relation = nodeMapping[bytes(node.int().tolist())]
        if isinstance(relation,list):
            for r in relation:
                example_ILP += f"{r}({problem_id},{node_id}).\n"
                bg_aleph.append(f"{r}({problem_id},{node_id}).\n")
        else:
            example_ILP += f"{relation}({problem_id},{node_id}).\n"
            bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")

    # create the edges
    edges = []
    for j in graph.edge_index.T.tolist():
        if j not in edges:
            edges.append(j)
    for j, edge in enumerate(edges):
        node_id1 = nodeIds[edge[0]]
        node_id2 = nodeIds[edge[1]]
        example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
        bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
    examples.append(example_ILP)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/KRK/FullDiag/krk.kb"
with open(file_path,'w') as f:
    for example in examples:
        f.write(example)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Aleph/FullDiag/exs.pl"
with open(file_path,'w') as f:
    for example in examples_aleph:
        f.write(example)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Aleph/FullDiag/bk.pl"
with open(file_path,'w') as f:
    for example in bg_aleph:
        f.write(example)


### Simple ###
""" 
No edge features and no node features (in the ILP syntax)
"""

dataset = KRKDataset(root='datasets/KRK/Simple')
datalist = dataset.data_list



examples = []
examples_aleph = []
bg_aleph = []
for i, graph in enumerate(datalist):
    example_ILP = ""
    problem_id = str(i)
    nodeIds = createNodeIds(graph.x.shape[0])

    # add the class example with the problem id and the truth label
    label = "pos" if graph.y.item() == 1 else "neg"
    example_ILP += f"krk({problem_id},{label}).\n"
    examples_aleph.append(f"{label}(krk({problem_id})).\n")

    # create the nodes
    for j, node in enumerate(graph.x):
        if j == 0:
            example_ILP += f"white_king({problem_id},{nodeIds[j]},{node[0]},{node[1]}).\n"
            bg_aleph.append(f"white_king({problem_id},{nodeIds[j]},{node[0]},{node[1]}).\n")
        elif j == 1:
            example_ILP += f"white_rook({problem_id},{nodeIds[j]},{node[0]},{node[1]}).\n"
            bg_aleph.append(f"white_rook({problem_id},{nodeIds[j]},{node[0]},{node[1]}).\n")
        elif j == 2:
            example_ILP += f"black_king({problem_id},{nodeIds[j]},{node[0]},{node[1]}).\n"
            bg_aleph.append(f"black_king({problem_id},{nodeIds[j]},{node[0]},{node[1]}).\n")

    # create the edges
    edges = []
    for j in graph.edge_index.T.tolist():
        if j not in edges:
            edges.append(j)
    for j, edge in enumerate(edges):
        node_id1 = nodeIds[edge[0]]
        node_id2 = nodeIds[edge[1]]
        example_ILP += f"edge({problem_id},{node_id1},{node_id2}).\n"
        bg_aleph.append(f"edge({problem_id},{node_id1},{node_id2}).\n")
    
    examples.append(example_ILP)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/KRK/Simple/krk.kb"
with open(file_path,'w') as f:
    for example in examples:
        f.write(example)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Aleph/Simple/exs.pl"
with open(file_path,'w') as f:
    for example in examples_aleph:
        f.write(example)

file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/KRK/Aleph/Simple/bk.pl"
with open(file_path,'w') as f:
    for example in bg_aleph:
        f.write(example)



