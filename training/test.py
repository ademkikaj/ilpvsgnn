import sys
import os

# Add the path to the datasets module to the PYTHONPATH environment variable
sys.path.append(os.path.abspath('../datasets/'))

from datasets.bongard_dataset import BongardDataset

dataset = BongardDataset()
