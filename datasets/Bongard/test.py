# perform unit testing on the Bongard dataset

import unittest
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from BongardDataset import BongardDataset

class TestBongardDataset(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.dataset = BongardDataset(root='datasets/Bongard/Graph')
        self.datasetHetero = BongardDataset(root='datasets/Bongard/Heterogeneous')


    
    def test_delete_datapoints(self):
        self.dataset.process()
        # remove 10 datapoints
        self.assertEqual(len(self.dataset), 392)
        self.dataset.remove_datapoints([0,1,2,3,4,5,6,7,8,9])
        self.assertEqual(len(self.dataset), 382)
        # reprocess the dataset from the original raw data
        self.dataset.process()
        self.assertEqual(len(self.dataset), 392)

    def test_len(self):
        self.dataset.process()
        self.assertEqual(len(self.dataset), 392)

        # remove 10 datapoints
        self.dataset.remove_datapoints([0,1,2,3,4,5,6,7,8,9])
        self.assertEqual(len(self.dataset), 382)

        self.assertEqual(len(self.dataset), len(self.dataset.data_list))

    # def test_shuffle(self):
    #     before_shuffle = self.dataset[0]
    #     random.shuffle(self.dataset.data_list)
    #     after_shuffle = self.dataset[0]
    #     self.assertNotEqual(True,False)
    
    def test_fold(self):
        self.dataset.process()
        kf = KFold(n_splits=5, shuffle=True,random_state=42)
        for train_index, test_index in kf.split(list(range(len(self.dataset)))):
            print("test: ", test_index[0])
    
    def test_split(self):
        self.dataset.process()
        train_index, test_index = train_test_split(list(range(len(self.dataset))), test_size=0.1,shuffle=False)
        print("test: ", test_index[0])
        print("train: ", train_index[0])
    
    def test_folds(self):
        self.dataset.process()
    
    def test_hetero(self):
        print(self.datasetHetero[0].metadata())
        


    
        
if __name__ == "__main__":
    unittest.main()