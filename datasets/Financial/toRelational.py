import pandas as pd
import os
import glob
import re

# load the tilde files
examples_path  = "datasets/Financial/Tilde/financial.kb"


# creating the relational dataframes
relations = {
    'financial_d': ['key','class'],
    'account': ['key','district_id','frequency','date'],
    'card': ['key','disp_id','card_type','date'],
    'client': ['key','client_id','birth','district_id'],
    'disp': ['key','disp_id','client_id','disp_type'],
    'district': ['district_id','district_name','region,inhabitants','mun1','mun2','mun3','mun4','cities','ratio','avgsal','unemploy95','unemploy96','enterpreneurs','crimes95','crimes96'],
    'loan': ['key','date','amount','duration','payments'],
    'order': ['key','bank_to','amount','symbol'],
    'trans': ['key','date','trans_type','operation','amount','balance','trans_char']
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
            print(line)


# write the relational dataframes to csv files
for pred in df.keys():
    df[pred].to_csv('datasets/Financial/Relational/'+pred+'.csv',index=False)