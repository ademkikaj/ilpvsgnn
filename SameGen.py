import os
from sklearn.model_selection import train_test_split
from Benchmark.benchmark import Benchmark
import torch
import pandas as pd
import yaml


benchmark = Benchmark()

dataset_name = "sameGen"
representations = ["node_only","node_edge",'edge_based','Klog']
#representations = ["node_edge"]
target = "class"
problem_id = "name1"
ilp_systems = ['tilde','aleph','popper']

TILDE = True
ALEPH = True
POPPER = False

GNN = False

ORIGINAL = False

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# print current directory

base_path = os.path.join("docker", "Benchmark", dataset_name)


# creating the directory structure -> can be done in the benchmark class
for repr in representations:
    path = os.path.join(base_path, repr)
    if not os.path.exists(path):
        os.makedirs(path)
    logic_path = os.path.join(path, "logic")
    graph_path = os.path.join(path, "graph")
    if not os.path.exists(logic_path):
        os.makedirs(logic_path)
    for ilp in ilp_systems:
        ilp_path = os.path.join(logic_path, ilp)
        if not os.path.exists(ilp_path):
            os.makedirs(ilp_path)
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
relational_path = os.path.join(base_path, "relational")
if not os.path.exists(relational_path):
    os.makedirs(relational_path)
relational_path_test = os.path.join(relational_path, "test")
if not os.path.exists(relational_path_test):
    os.makedirs(relational_path_test)
relational_path_train = os.path.join(relational_path, "train")
if not os.path.exists(relational_path_train):
    os.makedirs(relational_path_train)

results_path = os.path.join(base_path, "results")
if not os.path.exists(results_path):
    os.makedirs(results_path)

if not os.path.exists(os.path.join(base_path, "original")):
    os.makedirs(os.path.join(base_path, "original"))
if not os.path.exists(os.path.join(base_path, "original", "logic")):
    os.makedirs(os.path.join(base_path, "original", "logic"))
for ilp in ilp_systems:
    if not os.path.exists(os.path.join(base_path, "original", "logic", ilp)):
        os.makedirs(os.path.join(base_path, "original", "logic", ilp))


## split the data into train and test
    
#benchmark.new_split_relational_data(relational_path,dataset_name,target,problem_id)

#load the datasetName
df = pd.read_csv(os.path.join(relational_path, f"{dataset_name}.csv"))

# stratify the test set
train, test = train_test_split(df, test_size=0.21, stratify=df[target])
train.to_csv(os.path.join(relational_path, "train", f"{dataset_name}.csv"))
test.to_csv(os.path.join(relational_path, "test", f"{dataset_name}.csv"))



# create the name encoder

names = pd.read_csv(os.path.join(relational_path, "person.csv"))['name'].unique()

name_encoder = {}
for i,name in enumerate(names):
    name_encoder[name] = torch.zeros(len(names))
    name_encoder[name][i] = 1

name_decoder = {}
for i,name in enumerate(names):
    t = torch.zeros(len(names))
    t[i] = 1
    name_decoder[tuple(t.numpy())] = name



# conversions to different representations

import sys
from Benchmark.sameGen.toGraph import toGraph
from Benchmark.sameGen.toLogic import toLogic
from Benchmark.toILP import toILP



graph_converter = toGraph(relational_path=relational_path_train,dataset_name=dataset_name,dataset_problem_key=problem_id,target=target,name_encoder=name_encoder)
graph_converter_test = toGraph(relational_path=relational_path_test,dataset_name=dataset_name,dataset_problem_key=problem_id,target=target,name_encoder=name_encoder)

logic_converter = toLogic(dataset_name=dataset_name,name_decoder=name_decoder,relational_path=relational_path_train)
logic_converter_test = toLogic(dataset_name=dataset_name,name_decoder=name_decoder,relational_path=relational_path_test)


for repr in representations:

    # build the graph representations
    string = f"graph_converter.{repr}()"
    test_string = f"graph_converter_test.{repr}()"
    graphs = eval(string)
    graphs_test = eval(test_string)

    # write the graphs to the graph directory
    torch.save(graphs, os.path.join(base_path, repr, "graph", "train.pt"))
    torch.save(graphs_test, os.path.join(base_path, repr, "graph", "test.pt"))

    # convert the graphs to logic
    output_path = os.path.join(base_path, repr, "logic",dataset_name + ".kb")
    string = f"logic_converter.{repr}(graphs,'{output_path}')"
    eval(string)
    output_path_test = os.path.join(base_path, repr, "logic",dataset_name + "_test.kb")
    string = f"logic_converter_test.{repr}(graphs_test,'{output_path_test}')"
    eval(string)

    # remove the truth labels from the test file
    with open(output_path_test, "r") as file:
        lines = file.readlines()
    new_test = []
    for line in lines:
        if dataset_name not in line:
            new_test.append(line)
    with open(output_path_test, "w") as file:
        for line in new_test:
            file.write(line)
    



# Convert every logic representation to the ILP systems input format
            

    
for repr in representations:

    ilpConverter = toILP(relational_path=relational_path, logic_path=os.path.join(base_path, repr, "logic"), dataset_name=dataset_name)
    
    # Popper 
    if repr == "node_only":
        popper_bias = [
            "body_pred(parent,3).\n",
            "body_pred(child,3).\n",
            "body_pred(edge,3).\n",
            f"type({dataset_name},(id,name,name)).\n",
            "type(parent,(id,node_id,name)).\n",
            "type(child,(id,node_id,name)).\n",
            "type(edge,(id,node_id,node_id)).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+A,+B,+C,-D)).\n",
            "warmode(parent(+id,+-node_id,+-name)).\n",
            "warmode(child(+id,+-node_id,+-name)).\n",
            "warmode(person(+id,+-node_id,+-name)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,parent(+id, -name, -name)).\n",
            ":- modeb(*,child(+id,-node_id ,-name)).\n",
            ":- modeb(*,person(+id, -name, -name)).\n",
            ":- modeb(*,edge(+id, -name, -node_id)).\n",
            f":- determination({dataset_name}/3,parent/3).\n",
            f":- determination({dataset_name}/3,child/3).\n",
            f":- determination({dataset_name}/3,person/3).\n",
            f":- determination({dataset_name}/3,edge/3).\n",
        ]
    elif repr == "node_edge":
        popper_bias = [
            "body_pred(person,2).\n",
            "body_pred(parent,3).\n",
            "body_pred(same_gen,3).\n",
            f"type({dataset_name},(id,name,name)).\n",
            "type(parent,(id,name,name)).\n",
            "type(person,(id,name)).\n",
            "type(same_gen,(id,name,name)).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+A,+B,+C,-D)).\n",
            "warmode(person(+puzzle,-name)).\n",
            "warmode(parent(+puzzle,-name,-name)).\n",
            "warmode(same_gen(+puzzle,-name,-name)).\n",
            "auto_lookahead(person(Id,Name),[Name]).\n",
            "auto_lookahead(parent(Id,Parent,Name),[Parent,Name]).\n",
            "auto_lookahead(same_gen(Id,Name1,Name2),[Name1,Name2]).\n",
        ]
        aleph_settings = [
            ":- modeb(*,parent(+id, -name, -name)).\n",
            ":- modeb(*,person(+id, -name)).\n",
            ":- modeb(*,same_gen(+id, -name, -name)).\n",
            f":- determination({dataset_name}/3,parent/3).\n",
            f":- determination({dataset_name}/3,person/2).\n",
            f":- determination({dataset_name}/3,same_gen/3).\n",
        ]
    elif repr == "edge_based":
        popper_bias = [
            "body_pred(person,3).\n",
            "body_pred(parent,3).\n",
            "body_pred(same_gen,3).\n",
            "body_pred(edge,3).\n",
            "body_pred(instance,2).\n",
            f"type({dataset_name},(id,name,name)).\n",
            "type(parent,(id,node_id,node_id)).\n",
            "type(person,(id,node_id,name)).\n",
            "type(same_gen,(id,node_id,node_id)).\n",
            "type(edge,(id,node_id,node_id)).\n",
            "type(instance,(id,node_id)).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+A,+B,+C,-D)).\n",
            "warmode(person(+id,+-node_id,+-name)).\n",
            "warmode(parent(+id,+-node_id,+-name,+-name)).\n",
            "warmode(same_gen(+id,+-node_id,+-name,+-name)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
            "warmode(instance(+id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,parent(+id, -node_id, -name,-name)).\n",
            ":- modeb(*,person(+id, -node_id, -name)).\n",
            ":- modeb(*,same_gen(+id, -node_id, -node_id,-name)).\n",
            ":- modeb(*,edge(+id, -node_id, -node_id)).\n",

            ":- modeb(*,instance(+id, -node_id)).\n",

            f":- determination({dataset_name}/3,parent/3).\n",
            f":- determination({dataset_name}/3,person/3).\n",
            f":- determination({dataset_name}/3,same_gen/3).\n",
            f":- determination({dataset_name}/3,edge/3).\n",
            f":- determination({dataset_name}/3,instance/2).\n",
        ]
    elif repr == "Klog":
        popper_bias = [
            "body_pred(person,3).\n",
            "body_pred(parent,2).\n",
            "body_pred(same_gen,2).\n",
            "body_pred(edge,3).\n",
            f"type({dataset_name},(id,name,name)).\n",
            "type(person,(id,node_id,name)).\n",
            "type(parent,(id,node_id)).\n",
            "type(same_gen,(id,node_id)).\n",
            "type(edge,(id,node_id,node_id)).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+A,+B,+C,-D)).\n",
            "warmode(person(+id,+-node_id,+-name)).\n",
            "warmode(parent(+id,+-node_id)).\n",
            "warmode(same_gen(+id,+-node_id)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,parent(+id, -node_id)).\n",        
            ":- modeb(*,person(+id, -node_id, -name)).\n",
            ":- modeb(*,same_gen(+id, -node_id)).\n",
            ":- modeb(*,edge(+id, -node_id, -node_id)).\n",
            f":- determination({dataset_name}/3,parent/2).\n",
            f":- determination({dataset_name}/3,person/3).\n",
            f":- determination({dataset_name}/3,same_gen/2).\n",
            f":- determination({dataset_name}/3,edge/3).\n",
        ]
    
    ilpConverter.logicToPopper(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"), label = dataset_name ,bias_given=popper_bias)
    ilpConverter.logicToTilde(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),givesettings=tilde_settings)
    ilpConverter.logicToAleph(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),label= dataset_name,given_settings=aleph_settings)

if ORIGINAL:
    # original format
    ilpConverter = toILP(relational_path=relational_path, logic_path=os.path.join(base_path, "original", "logic"), dataset_name=dataset_name)
    # convert the train_data
    ilpConverter.to_kb(os.path.join(base_path, "original", "logic","sameGen.kb"),test=False)
    # convert the test data
    ilpConverter.to_kb(os.path.join(base_path, "original", "logic","sameGen_test.kb"),test=True)
    output_path_test = os.path.join(base_path, "original", "logic","sameGen_test.kb")
    # remove the truth labels from the test file
    with open(output_path_test, "r") as file:
        lines = file.readlines()
    new_test = []
    for line in lines:
        if dataset_name not in line:
            new_test.append(line)
    with open(output_path_test, "w") as file:
        for line in new_test:
            file.write(line)
    # Popper
    popper_bias = [
        "head_pred(sameGen,2).\n",
        "body_pred(same_gen,2).\n",
        "body_pred(person,1).\n",
        "body_pred(parent,2).\n"
    ]
    ilpConverter.toPopper(popper_bias)
    from Benchmark.popper_system import Popper
    popper = Popper(name=dataset_name,relational_path=relational_path,target=target)
    popper_results = popper.run(representation_path=os.path.join(base_path,"original"))
    popper_results['representation'] = repr
    print(popper_results)
    # TILDE
    tilde_settings = [
        "predict(sameGen(+A,+B,-C)).\n",
        "warmode(same_gen(+A,-B)).\n",
        "warmode(person(+A)).\n",
        "warmode(parent(+A,-B)).\n"
    ]
    ilpConverter.toTilde(tilde_settings)
    from Benchmark.tilde import Tilde
    from Benchmark.aleph_system import Aleph
    tilde = Tilde(dataset_name=dataset_name, relational_path=relational_path, target=target)
    tilde_results = tilde.run(tilde_input_path=os.path.join(base_path,"original"))
    tilde_results['representation'] = repr
    print(tilde_results)
    # aleph
    aleph_settings = [
        ":- modeh(1,sameGen(+name,+name)).\n",
        ":- modeb(*,same_gen(+name,-name)).\n",
        ":- modeb(*,person(+name)).\n",
        ":- modeb(*,parent(+name,-name)).\n",
        ":- determination(sameGen/2,same_gen/2).\n",
        ":- determination(sameGen/2,person/1).\n",
        ":- determination(sameGen/2,parent/2).\n"
    ]

    ilpConverter.toAleph(aleph_settings)

    aleph = Aleph(name=dataset_name, relational_path=relational_path, target=target)
    aleph_results = aleph.run(representation_path=os.path.join(base_path,"original"))
    print(aleph_results)


# All the files are in the correct locations for the logic and graph representations
# Run the ILP systems on the different representations
from Benchmark.tilde import Tilde
from Benchmark.popper_system import Popper
from Benchmark.aleph_system import Aleph

for repr in representations:
    print("Representation: ",repr)
    results = pd.DataFrame()

    if TILDE:
        tilde = Tilde(dataset_name=dataset_name, relational_path=relational_path, target=target)
        tilde_results = tilde.run(tilde_input_path=os.path.join(base_path,repr))
        tilde_results['representation'] = repr
        results = pd.concat([results,tilde_results])
    if POPPER:
        popper = Popper(name=dataset_name,relational_path=relational_path,target=target)
        popper_results = popper.run(representation_path=os.path.join(base_path,repr))
        popper_results['representation'] = repr
        results = pd.concat([results,popper_results])
    if ALEPH:
        aleph = Aleph(name=dataset_name, relational_path=relational_path,target=target)
        aleph_results = aleph.run(representation_path=os.path.join(base_path,repr))
        aleph_results['representation'] = repr
        results = pd.concat([results,aleph_results])

    results.to_csv(os.path.join(base_path,"results",f"results_logic_{repr}.csv"),index=False)


#merge the results representations
total_results = pd.DataFrame()
for repr in representations:
    results = pd.read_csv(os.path.join(base_path,"results",f"results_logic_{repr}.csv"))
    os.remove(os.path.join(base_path,"results",f"results_logic_{repr}.csv"))
    total_results = pd.concat([total_results,results])
total_results.to_csv(os.path.join(base_path,"results","results_logic.csv"),index=False)

print(total_results)


# Run the GNN's on the different representations
if GNN:
    with open("Benchmark/gnn_config.yaml") as file:
        config = yaml.safe_load(file)

    total_runs = len(config['models']) * len(config['layers']) * len(config['hidden_dims']) * len(representations) * config['repetitions']
    done_runs = 0
    total_gnn_data = pd.DataFrame()
    for model in config['models']:
        for layers in config['layers']:
            for hidden_dims in config['hidden_dims']:
                for representation in representations:
                    print(f"Run {done_runs}/{total_runs}")
                    graphs = torch.load(os.path.join(base_path,representation,"graph","train.pt"))
                    test_graphs = torch.load(os.path.join(base_path,representation,"graph","test.pt"))
                    result = benchmark.run_gnn(graphs,test_graphs,model,layers,hidden_dims,config)
                    result['representation'] = representation
                    total_gnn_data = pd.concat([total_gnn_data,result])
                    done_runs += config['repetitions']

    total_gnn_data.to_csv(os.path.join(base_path,"results","results_gnn.csv"),index=False)

