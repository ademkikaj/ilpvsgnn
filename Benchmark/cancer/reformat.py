import pandas as pd
import os

base_path = "Benchmark/cancer/Relational"
# get all filenames in the relational folder
all_files = os.listdir(base_path)
# get all filenames starting with "atm"
atm_files = [filename for filename in all_files if filename.startswith("atm")]
# get all filenames starting with "sbond"
sbond_files = [filename for filename in all_files if filename.startswith("sbond")]


atom_df = pd.DataFrame(columns=["id","atom_id","atom_type","charge"])

for filename in atm_files:
    df = pd.read_csv(os.path.join(base_path,filename))
    df.rename(columns={"drug":"id","atomid":"atom_id","atomtype":"atom_type"},inplace=True)
    atom_df = pd.concat([atom_df,df],ignore_index=True)

atom_df.to_csv(os.path.join(base_path,"atom.csv"),index=False)

bond_df = pd.DataFrame(columns=["id","atom_id_1","atom_id_2","bond_type"])

for filename in sbond_files:
    df = pd.read_csv(os.path.join(base_path,filename))
    df["bond_type"] = int(filename[-5])
    df.rename(columns={"drug":"id","atomid1":"atom_id_1","atomid2":"atom_id_2"},inplace=True)
    bond_df = pd.concat([bond_df,df],ignore_index=True)

bond_df.to_csv(os.path.join(base_path,"bond.csv"),index=False)