import pandas as pd
import os
import glob
import re

class toRelational:
    
    def __init__(self,logic_path,output_path,relations) -> None:
        self.logic_path = logic_path
        self.output_path = output_path
        self.relations = relations
    
    def logic_to_relational(self,):

        with open(self.logic_path,'r') as file:
            lines = file.readlines()
        df = {pred: pd.DataFrame(columns=self.relations[pred]) for pred in self.relations.keys()}

        pattern = r"(\w+)\(([^)]+)\)"
        for line in lines:
            match = re.match(pattern,line)
            if match:
                predicate = match.group(1)
                args = match.group(2).split(',')

                new_row = pd.DataFrame([{key:val for key,val in zip(self.relations[predicate],args)}])
                df[predicate] = pd.concat([df[predicate],new_row],ignore_index=True)
        
        for pred in df.keys():
            df[pred].to_csv(self.output_path+pred+'.csv',index=False)



# # load relational csv files into a dictionary of format {relation: dataframe}
# path = '/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Relational'
# all_files = glob.glob(os.path.join(path, "*.csv"))
# kb = {}
# predicates = []
# for filename in all_files:
#     df = pd.read_csv(filename)
#     print(filename)
#     relation = filename.split('/')[-1].split('.')[0]
#     predicates.append(relation)
#     kb[relation] = df



# amount_of_examples = 50
# amount_of_positives = 25
# amount_of_negatives = 25

# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/docker/Bongard/NoModels50/bongard.kb"
# with open(file_path,'w') as file:
#     pos_examples = kb['bongard'][kb['bongard']['class'] == 'pos'].sample(amount_of_positives)
#     neg_examples = kb['bongard'][kb['bongard']['class'] == 'neg'].sample(amount_of_negatives)
#     examples = pd.concat([pos_examples,neg_examples])
#     for i in range(len(examples)):
#         bongard_id = examples.iloc[i]['problemId']
#         squares = kb['square'][kb['square']['problemId'] == bongard_id]
#         circles = kb['circle'][kb['circle']['problemId'] == bongard_id]
#         triangles = kb['triangle'][kb['triangle']['problemId'] == bongard_id]
#         configs = kb['config'][kb['config']['problemId'] == bongard_id]
#         ins = kb['in'][kb['in']['problemId'] == bongard_id]
#         bongard_id = str(bongard_id)

#         file.write('\n')
#         file.write("bongard("+bongard_id+","+examples.iloc[i]["class"] +").\n")
    
#         for j in range(len(squares)):
#             file.write("square("+bongard_id+","+str(squares.iloc[j]['objectId'])+").\n")
#         for j in range(len(circles)):
#             file.write("circle("+bongard_id+","+str(circles.iloc[j]['objectId'])+").\n")
#         for j in range(len(triangles)):
#             file.write("triangle("+bongard_id+","+str(triangles.iloc[j]['objectId'])+").\n")
#         for j in range(len(configs)):
#             file.write("config("+bongard_id+","+str(configs.iloc[j]['objectId'])+","+str(configs.iloc[j]['orient'])+").\n")
#         for j in range(len(ins)):
#             file.write("in("+bongard_id+","+str(ins.iloc[j]['objectId1'])+","+str(ins.iloc[j]['objectId2'])+").\n")


# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Popper/exs.pl"

# with open(file_path,'w') as file:
#     pos_examples = kb['bongard'][kb['bongard']['class'] == 'pos']
#     neg_examples = kb['bongard'][kb['bongard']['class'] == 'neg']
#     examples = pd.concat([pos_examples,neg_examples])
#     for i in range(len(examples)):
#         bongard_id = examples.iloc[i]['problemId']
#         if kb['bongard'][kb['bongard']['problemId'] == bongard_id]['class'].iloc[0] == 'pos':
#             file.write("pos(bongard("+str(bongard_id)+")).\n")
#         else:
#             file.write("neg(bongard("+str(bongard_id)+")).\n")


# file_path = "/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/Bongard/Popper/bk.pl"
# with open(file_path,'w') as file:
#     for i in range(len(examples)):
#         bongard_id = examples.iloc[i]['problemId']
#         squares = kb['square'][kb['square']['problemId'] == bongard_id]
#         circles = kb['circle'][kb['circle']['problemId'] == bongard_id]
#         triangles = kb['triangle'][kb['triangle']['problemId'] == bongard_id]
#         configs = kb['config'][kb['config']['problemId'] == bongard_id]
#         ins = kb['in'][kb['in']['problemId'] == bongard_id]

#         bongard_id = str(bongard_id)

#         for j in range(len(squares)):
#             file.write("square("+bongard_id+',' +str(squares.iloc[j]['objectId'])+").\n")
#         for j in range(len(circles)):
#             file.write("circle("+bongard_id+',' +str(circles.iloc[j]['objectId'])+").\n")
#         for j in range(len(triangles)):
#             file.write("triangle("+bongard_id+',' +str(triangles.iloc[j]['objectId'])+").\n")
#         for j in range(len(configs)):
#             file.write("config("+bongard_id+',' +str(configs.iloc[j]['objectId'])+','+str(configs.iloc[j]['orient'])+").\n")
#         for j in range(len(ins)):
#             file.write("in("+bongard_id+',' +str(ins.iloc[j]['objectId1'])+','+str(ins.iloc[j]['objectId2'])+").\n")
    


# # let's replicate the model format 