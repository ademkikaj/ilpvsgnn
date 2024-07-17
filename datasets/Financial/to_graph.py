import pandas as pd
import glob
import os


# load relational csv files into dictionary of format {relation : dataframe}
### TO DO
# improve this to use joins?
path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Financial/Relational"
all_files = glob.glob(os.path.join(path, "*.csv"))
kb = {}
predicates = []
for filename in all_files:
    relation = filename.split("/")[-1].split(".")[0]
    predicates.append(relation)
    kb[relation] = pd.read_csv(filename, header=None)

print(kb.keys())