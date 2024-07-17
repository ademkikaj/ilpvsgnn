import pandas as pd
import os
import glob
import re

# atoms and bonds

# atom(drug_id, atom_id, atom_type,charge)
# bond(drug_id, atom_id_1, atom_id_2, bond_type)

relations = {
    'atom': ['drug_id','atom_id','atom_type','charge'],
    'bond': ['drug_id','atom_id_1','atom_id_2','bond_type'],
    'drug': ['drug_id','ind1','inda','logp','lumo','active'],
    'benzene' : ['drug_id','atom_id_1','atom_id_2','atom_id_3','atom_id_4','atom_id_5','atom_id_6'],
    'nitro' : ['drug_id','atom_id_1','atom_id_2','atom_id_3','atom_id_4'],
}




ind1 = {}
file_path = "datasets/Muta/Logic/ind1.pl"
with open(file_path, 'r') as file:
    lines = file.readlines()
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')
            ind1[args[0]] = args[1]
inda = {}
file_path = "datasets/Muta/Logic/inda.pl"
with open(file_path, 'r') as file:
    lines = file.readlines()
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')
            inda[args[0]] = args[1]
logp = {}
file_path = "datasets/Muta/Logic/logp.pl"
with open(file_path, 'r') as file:
    lines = file.readlines()
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')
            logp[args[0]] = args[1]

lumo = {}
file_path = "datasets/Muta/Logic/lumo.pl"
with open(file_path, 'r') as file:
    lines = file.readlines()
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')
            lumo[args[0]] = args[1]
active = {}
file_path = "datasets/Muta/Logic/exs.pl"
with open(file_path, 'r') as file:
    lines = file.readlines()
    pattern = r"(\w+)\((\w+)\(([^)]+)\)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            clas = match.group(1)
            arg = match.group(3)
            active[arg] = clas
        

# combine into one dataframe

drug = []
for drug_id in ind1.keys():
    if drug_id == "d189":
        break
    drug.append({'drug_id': drug_id, 'ind1': ind1[drug_id], 'inda': inda[drug_id], 'logp': logp[drug_id], 'lumo': lumo[drug_id], 'active': active[drug_id]})
drug = pd.DataFrame(drug)
drug.to_csv('datasets/Muta/Relational/drug.csv', index=False)



atom = []
bond = []
file_path = "datasets/Muta/Logic/atom_bond.pl"
with open(file_path, 'r') as file:
    lines = file.readlines()
    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')
            
            if predicate == 'atm':
                atom.append({'drug_id': args[0], 'atom_id': args[1], 'atom_type': args[2], 'charge': args[3]})
            elif predicate == 'bond':
                bond.append({'drug_id': args[0], 'atom_id_1': args[1], 'atom_id_2': args[2], 'bond_type': args[3]})

atom  = pd.DataFrame(atom)
bond  = pd.DataFrame(bond)

# save to csv
atom.to_csv('datasets/Muta/Relational/atom.csv', index=False)
bond.to_csv('datasets/Muta/Relational/bond.csv', index=False)

    


ring_struc = {}
ring_struc_prevelance = {}
ring_struc_predicates = []
filepath = "datasets/Muta/Logic/ring_struc.pl"
benzene = []
nitro = []

with open(filepath, 'r') as file:
    lines = file.readlines()

    pattern = r"(\w+)\(([^)]+)\)"

    for line in lines:
        match = re.match(pattern,line)
        if match:
            predicate = match.group(1)
            args = match.group(2).split(',')
            drug_id = args[0]
            if predicate not in ring_struc_predicates:
                ring_struc_predicates.append(predicate)
                ring_struc[predicate] = len(args)
                ring_struc_prevelance[predicate] = 1
            ring_struc_prevelance[predicate] += 1

            args = [a.replace('[','').replace(']','') for a in args]

            if predicate == 'benzene':
                i = len(benzene)
                benzene.append({'drug_id': drug_id, 'atom_id_1': args[1], 'atom_id_2': args[2], 'atom_id_3': args[3], 'atom_id_4': args[4], 'atom_id_5': args[5], 'atom_id_6': args[6]})
            elif predicate == 'nitro':
                nitro.append({'drug_id': drug_id, 'atom_id_1': args[1], 'atom_id_2': args[2], 'atom_id_3': args[3], 'atom_id_4': args[4]})
            

benzene  = pd.DataFrame(benzene)
nitro  = pd.DataFrame(nitro)
print(ring_struc_predicates)
print(ring_struc_prevelance)


# save to csv
benzene.to_csv('datasets/Muta/Relational/benzene.csv', index=False)
nitro.to_csv('datasets/Muta/Relational/nitro.csv', index=False)


