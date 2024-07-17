from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import pandas as pd
import torch

### check the mutag TU dataset

mutag = TUDataset(root='/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta', name='MUTAG')

atoms = pd.read_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Relational/atom.csv')
bonds = pd.read_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Relational/bond.csv')
drugs = pd.read_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Relational/drug.csv')

benzenes = pd.read_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Relational/benzene.csv')
nitro = pd.read_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Muta/Relational/nitro.csv')


# print(bonds['bond_type'].unique())
# print(atoms['atom_type'].value_counts())
# print(atoms[atoms['atom_type'] == 'i']["drug_id"])
# print(atoms[atoms['atom_type'] == 's']["drug_id"])

# normalize the drug columns lumo and logp, maximum absolut scaling
drugs['logp'] = drugs['logp']/drugs['logp'].abs().max()
drugs['lumo'] = drugs['lumo']/drugs['lumo'].abs().max()


### create the DrugNode dataset
edge_attr_emb = {
    1 : torch.tensor([1,0,0,0,0,0],dtype = torch.float),
    2 : torch.tensor([0,1,0,0,0,0],dtype = torch.float),
    3 : torch.tensor([0,0,1,0,0,0],dtype = torch.float),
    4 : torch.tensor([0,0,0,1,0,0],dtype = torch.float),
    5 : torch.tensor([0,0,0,0,1,0],dtype = torch.float),
    7 : torch.tensor([0,0,0,0,0,1],dtype = torch.float)
}
node_emb = {
    'c' : torch.tensor([1,0,0,0,0,0,0,0],dtype=torch.float),
    'n' : torch.tensor([0,1,0,0,0,0,0,0],dtype=torch.float),
    'o' : torch.tensor([0,0,1,0,0,0,0,0],dtype=torch.float),
    'h' : torch.tensor([0,0,0,1,0,0,0,0],dtype=torch.float),
    'cl': torch.tensor([0,0,0,0,1,0,0,0],dtype=torch.float),
    'f' : torch.tensor([0,0,0,0,0,1,0,0],dtype=torch.float),
    'br': torch.tensor([0,0,0,0,0,0,1,0],dtype=torch.float),
    'i' : torch.tensor([0,0,0,0,0,0,0,1],dtype=torch.float)
}

datalist = []
for i in range(len(drugs)):

    atom_id_to_index = {}

    drug_id = drugs.iloc[i]["drug_id"]

    curr_atoms = atoms[atoms['drug_id'] == drug_id]
    curr_bonds = bonds[bonds['drug_id'] == drug_id]
    edge_indices = [[],[]]

    x = torch.empty((len(curr_atoms)+1,12), dtype=torch.float)
    edge_attributes = []

    # create nodes
    for j in range(len(curr_atoms)):
        atom = curr_atoms.iloc[j]
        atom_id_to_index[atom['atom_id']] = j
        x[j] = torch.cat((node_emb[atom['atom_type']],torch.zeros(4,dtype=torch.float)))

    # create edges and node attributes
    for j in range(len(curr_bonds)):
        bond = curr_bonds.iloc[j]
        edge_attr = edge_attr_emb[bond['bond_type']]
        
        atom_id_1 = bond['atom_id_1']
        index_1 = atom_id_to_index[atom_id_1]
        atom_id_2 = bond['atom_id_2']
        index_2 = atom_id_to_index[atom_id_2]
        edge_indices[0].append(index_1)
        edge_indices[1].append(index_2)
        edge_attributes.append(edge_attr)
        edge_indices[0].append(index_2)
        edge_indices[1].append(index_1)
        edge_attributes.append(edge_attr)

    # truth value
    value = 1 if drugs.iloc[i]['active'] == 'pos' else 0
    y = torch.tensor([value],dtype=torch.int64)

    # add the graph info to the x 
    ind1 = drugs.iloc[i]['ind1']
    inda = drugs.iloc[i]['inda']
    logp = drugs.iloc[i]['logp']
    lumo = drugs.iloc[i]['lumo']
    x[-1] = torch.concatenate((torch.zeros(8),torch.tensor([ind1,inda,logp,lumo],dtype=torch.float)))
    # add the indices
    for j in range(x.shape[0]-1):
        edge_indices[0].append(j)
        edge_indices[1].append(x.shape[0]-1)
        edge_attributes.append(torch.tensor([0,0,0,0,0,0],dtype=torch.float))
        edge_indices[0].append(x.shape[0]-1)
        edge_indices[1].append(j)
        edge_attributes.append(torch.tensor([0,0,0,0,0,0],dtype=torch.float))
    


    edge_indices = torch.tensor(edge_indices,dtype=torch.int64)
    edge_attris = torch.empty((len(edge_attributes),6),dtype=torch.float)
    for i in range(len(edge_attributes)):
        edge_attris[i] = edge_attributes[i]

    graph = Data(x=x,edge_index=edge_indices,edge_attr=edge_attris,y=y)
    datalist.append(graph)



torch.save(datalist,'datasets/Muta/DrugNode/raw/datalist.pt')
    



print(datalist[0].edge_index)

print([max(datalist[i].edge_index[0]) for i in range(len(datalist))])