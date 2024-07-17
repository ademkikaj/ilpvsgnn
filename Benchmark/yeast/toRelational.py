import pandas as pd
import os
import glob
import re

# load the tilde files
examples_path  = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Yeast/yeast.kb"

# creating the relational dataframes
relations = {
    'yeast': ['gene','class'],
    'complex': ['gene','complex'],
    'enzyme': ['gene','enzyme'],
    'interaction': ['gene1','gene2','intertype'],
    'location': ['gene','location'],
    'path': ['gene1','gene2'],
    'phenotype': ['gene','phenotype'],
    'protein_class': ['gene','class'],
    'rcomplex': ['gene','complex'],
    'renzyme': ['gene','enzyme'],
    'rphenotype': ['gene','phenotype'],
    'rprotein_class': ['gene','class']
}

df = {pred: pd.DataFrame(columns=relations[pred]) for pred in relations.keys()}

with open(examples_path, 'r') as file:
    lines = file.readlines()
    
    # Declare the relation pattern
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')

            new_row = pd.DataFrame([{key:val for key,val in zip(relations[predicate],args)}])
            df[predicate] = pd.concat([df[predicate],new_row],ignore_index=True)
        else:
            print("No match line: ",line)


# write the relational dataframes to csv files
for pred in df.keys():
    df[pred].to_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/Benchmark/yeast/Relational/'+pred+'.csv',index=False)