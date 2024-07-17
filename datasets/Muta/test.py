from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='datasets/Muta/MUTAG', name='MUTAG')
print(dataset[0].x[0])