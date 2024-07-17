import unittest
from KRKDataset import KRKDataset
import torch
from torch_geometric.loader import DataLoader

class TestKRKDataset(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dataset = KRKDataset(root='datasets/KRK/FullBoard')

    
    def test_shuffle(self):
        torch.manual_seed(50)
        self.dataset = self.dataset[torch.randperm(len(self.dataset))]
        test_dataset = self.dataset[650:]
        self.dataset.remove_datapoints(torch.arange(650,len(self.dataset),1))

    def test_loader(self):
        loader = DataLoader(self.dataset.data_list, batch_size=1, shuffle=True)
        for batch in loader:
            print(batch)
            
        


        
    
    
    


if __name__ == "__main__":
    unittest.main()