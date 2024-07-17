import os 
import pandas as pd
import numpy as np
import json

base_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Benchmark/financial/relational"
files_names = ["account.csv", "card.csv", "client.csv", "disp.csv", "district.csv", "loan.csv", "order.csv", "trans.csv"]
file_paths = [os.path.join(base_path, file_name) for file_name in files_names]

all_encoders = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)[:-4]
    df = pd.read_csv(file_path)
    encoders = {}
    for column in df.columns:
        if (df[column].dtype == "object" or not np.issubdtype(df[column].dtype, np.number)) and "id" not in column.lower():
            encoders[column] = {value: index for index, value in enumerate(df[column].unique())}
    all_encoders[file_name] = encoders


# from these encoders create the decoders as well
all_decoders = {}
for file_name, encoders in all_encoders.items():
    decoders = {}
    for column, encoder in encoders.items():
        decoders[column] = {index: value for value, index in encoder.items()}
    all_decoders[file_name] = decoders



with open("Benchmark/financial/encoders.json", "w") as f:
    json.dump(all_encoders, f,indent=4)

with open("Benchmark/financial/decoders.json", "w") as f:
    json.dump(all_decoders, f,indent=4)