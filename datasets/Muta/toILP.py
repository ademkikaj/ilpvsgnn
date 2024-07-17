from MutaDataset import MutaDataset
from torch_geometric.datasets import TUDataset
import torch

def createNodeIds(n):
    return {i:'n'+str(i) for i in range(n)}

### TUDataset ###

dataset = TUDataset(root='datasets/Muta/MUTAG', name='MUTAG')
print("amount of positives: ", len([data for data in dataset if torch.equal(data.y,torch.tensor([1]))]))
print("amount of negatives: ", len([data for data in dataset if torch.equal(data.y,torch.tensor([0]))]))

nodeMapping = {
    bytes([1,0,0,0,0,0,0]): "c",
    bytes([0,1,0,0,0,0,0]): "n",
    bytes([0,0,1,0,0,0,0]): "o",
    bytes([0,0,0,1,0,0,0]): "f",
    bytes([0,0,0,0,1,0,0]): "i",
    bytes([0,0,0,0,0,1,0]): "cl",
    bytes([0,0,0,0,0,0,1]): "br"
}

edgeMapping = {
    bytes([1,0,0,0]) : "aromatic",
    bytes([0,1,0,0]) : "single",
    bytes([0,0,1,0]) : "double",
    bytes([0,0,0,1]) : "triple"
}

examples = []
examples_aleph = []
bg_aleph = []
for i,graph in enumerate(dataset):
    example_ILP = ""
    problem_id = str(i)
    nodeIds = createNodeIds(graph.x.shape[0])

    # add the class example with the problem id and the truth label
    if graph.y.item() == 1:
        label = "pos"
    else:
        label = "neg"
    example_ILP += f"muta({problem_id},{label}).\n"
    examples_aleph.append(f"{label}(muta({problem_id})).\n")
    
    # create the nodes
    for j,node in enumerate(graph.x):
        node_id = nodeIds[j]
        relation = nodeMapping[bytes(node.int().tolist())]
        example_ILP += f"{relation}({problem_id},{node_id}).\n"
        bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
    # create the edges
    edges = []
    for j in graph.edge_index.T.tolist():
        if j not in edges:
            edges.append(j)
    for j,edge in enumerate(edges):
        node_id1 = nodeIds[edge[0]]
        node_id2 = nodeIds[edge[1]]
        edge_attr = edgeMapping[bytes(graph.edge_attr[j].int().tolist())]
        example_ILP += f"{edge_attr}({problem_id},{node_id1},{node_id2}).\n"
        bg_aleph.append(f"{edge_attr}({problem_id},{node_id1},{node_id2}).\n")
    
    examples.append(example_ILP)

file_path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Muta/TUDataset/muta.kb'
with open(file_path,'w') as file:
    for example in examples:
        file.write(example)

file_path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Aleph/TUDataset/exs.pl'
with open(file_path,'w') as file:
    for example in examples_aleph:
        file.write(example)
file_path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Aleph/TUDataset/bk.pl'
with open(file_path,'w') as file:
    for example in bg_aleph:
        file.write(example)

### DrugNode ###

dataset = MutaDataset(root='datasets/Muta/DrugNode')
datalist = dataset.data_list
print("amount of positives: ", len([data for data in datalist if torch.equal(data.y,torch.tensor([1]))]))
print("amount of negatives: ", len([data for data in datalist if torch.equal(data.y,torch.tensor([0]))]))

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
examples_aleph = []
bg_aleph = []

for i,graph in enumerate(dataset):
    example_ILP = ""
    problem_id = str(i)
    nodeIds = createNodeIds(graph.x.shape[0]-1)

    # add the class example with the problem id and the truth label
    if graph.y.item() == 1:
        label = "pos"
    else:
        label = "neg"
    example_ILP += f"muta({problem_id},{label}).\n"
    examples_aleph.append(f"{label}(muta({problem_id})).\n")
    
    # create the nodes
    for j,node in enumerate(graph.x):
        first_node = node[:8]
        relation = nodeMapping[bytes(first_node.int().tolist())]
        if relation == 'drug':
            ind1 = node[8].item()
            inda = node[9].item()
            logp = node[10].item()
            lumo = node[11].item()
            example_ILP += f"{relation}({problem_id},{ind1},{inda},{logp},{lumo}).\n"
            bg_aleph.append(f"{relation}({problem_id},{ind1},{inda},{logp},{lumo}).\n")
        else:
            node_id = nodeIds[j]
            example_ILP += f"{relation}({problem_id},{node_id}).\n"
            bg_aleph.append(f"{relation}({problem_id},{node_id}).\n")
    
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
        example_ILP += f"{edge_attr}({problem_id},{node_id1},{node_id2}).\n"
        bg_aleph.append(f"{edge_attr}({problem_id},{node_id1},{node_id2}).\n")
    
    examples.append(example_ILP)

file_path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Muta/DrugNode/muta.kb'
with open(file_path,'w') as file:
    for example in examples:
        file.write(example)

file_path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Aleph/DrugNode/exs.pl'
with open(file_path,'w') as file:
    for example in examples_aleph:
        file.write(example)
file_path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Aleph/DrugNode/bk.pl'
with open(file_path,'w') as file:
    for example in bg_aleph:
        file.write(example)