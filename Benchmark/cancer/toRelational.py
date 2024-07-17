import pandas as pd
import os
import glob
import re

# load the tilde files
examples_path  = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Cancer/cancer.kb"

# creating the relational dataframes
relations = {
    "canc_d":["drug","class"],
    "atm_as":["drug","atomid","atomtype","charge"],
    "atm_ba":["drug","atomid","atomtype","charge"],
    "atm_br":["drug","atomid","atomtype","charge"],
    "atm_c":["drug","atomid","atomtype","charge"],
    "atm_ca":["drug","atomid","atomtype","charge"],
    "atm_cl":["drug","atomid","atomtype","charge"],
    "atm_cu":["drug","atomid","atomtype","charge"],
    "atm_f":["drug","atomid","atomtype","charge"],
    "atm_h":["drug","atomid","atomtype","charge"],
    "atm_hg":["drug","atomid","atomtype","charge"],
    "atm_i":["drug","atomid","atomtype","charge"],
    "atm_k":["drug","atomid","atomtype","charge"],
    "atm_mn":["drug","atomid","atomtype","charge"],
    "atm_n":["drug","atomid","atomtype","charge"],
    "atm_na":["drug","atomid","atomtype","charge"],
    "atm_o":["drug","atomid","atomtype","charge"],
    "atm_p":["drug","atomid","atomtype","charge"],
    "atm_pb":["drug","atomid","atomtype","charge"],
    "atm_s":["drug","atomid","atomtype","charge"],
    "atm_se":["drug","atomid","atomtype","charge"],
    "atm_sn":["drug","atomid","atomtype","charge"],
    "atm_te":["drug","atomid","atomtype","charge"],
    "atm_ti":["drug","atomid","atomtype","charge"],
    "atm_zn":["drug","atomid","atomtype","charge"],
    "sbond_1":["drug","atomid","atomid"],
    "sbond_2":["drug","atomid","atomid"],
    "sbond_3":["drug","atomid","atomid"],
    "sbond_7":["drug","atomid","atomid"],
    "kmap":["drug","atomid","atomtype","charge"]
}

df = {pred:[]  for pred in relations.keys()}


with open(examples_path, 'r') as file:
    lines = file.readlines()
    
    # Declare the relation pattern
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')

            # new_row = pd.DataFrame([{key:val for key,val in zip(relations[predicate],args)}])
            df[predicate] += [[val for val in args]]
            #df[predicate] = pd.concat([df[predicate],new_row],ignore_index=True)
        else:
            print("No match line: ",line)

for pred in df.keys():
    df[pred] = pd.DataFrame(df[pred],columns=relations[pred])

# write the relational dataframes to csv files
for pred in df.keys():
    df[pred].to_csv('/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/Benchmark/cancer/Relational/'+pred+'.csv',index=False)

    