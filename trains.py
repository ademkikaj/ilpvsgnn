import os
from sklearn.model_selection import train_test_split
from Benchmark.benchmark import Benchmark
import torch
import pandas as pd
import yaml

benchmark = Benchmark()

dataset_name = "train"
representations = ["node_only","node_edge",'edge_based','Klog']
#representations = ["node_only","node_edge","Klog"]
target = "class"
problem_id = "id"
ilp_systems = ['tilde','aleph','popper']



TILDE = True
ALEPH = False
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
    
benchmark.split_relational_data(relational_path, dataset_name,class_name=target,problem_id=problem_id)

df_train = pd.read_csv(os.path.join(relational_path_train, dataset_name + ".csv"))
df_train = df_train.sort_values(by=[problem_id])
df_train.to_csv(os.path.join(relational_path_train, dataset_name + ".csv"), index=False)
df_test = pd.read_csv(os.path.join(relational_path_test, dataset_name + ".csv"))
df_test = df_test.sort_values(by=[problem_id])
df_test.to_csv(os.path.join(relational_path_test, dataset_name + ".csv"), index=False)


# conversions to different representations

import sys
from Benchmark.train.toGraph import toGraph
from Benchmark.train.toLogic import toLogic
from Benchmark.toILP import toILP

graph_converter = toGraph(relational_path=relational_path_train,dataset_name=dataset_name,dataset_problem_key=problem_id,target=target)
graph_converter_test = toGraph(relational_path=relational_path_test,dataset_name=dataset_name,dataset_problem_key=problem_id,target=target)

logic_converter = toLogic(dataset_name=dataset_name,relational_path=relational_path_train,problem_key=problem_id)


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
    string = f"logic_converter.{repr}(graphs_test,'{output_path_test}')"
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
            "body_pred(short,2).\n",
            "body_pred(long,2).\n",
            "body_pred(two_wheels,2).\n",
            "body_pred(three_wheels,2).\n",
            "body_pred(roof_open,2).\n",
            "body_pred(roof_closed,2).\n",
            "body_pred(zero_load,2).\n",
            "body_pred(one_load,2).\n",
            "body_pred(two_load,2).\n",
            "body_pred(three_load,2).\n",
            "body_pred(circle,2).\n",
            "body_pred(triangle,2).\n",
            "body_pred(edge,3).\n",
            "type(train,(id,)).\n",
            "type(short,(id,car_id)).\n",
            "type(long,(id,car_id)).\n",
            "type(two_wheels,(id,car_id)).\n",
            "type(three_wheels,(id,car_id)).\n",
            "type(roof_open,(id,car_id)).\n",
            "type(roof_closed,(id,car_id)).\n",
            "type(zero_load,(id,load_id)).\n",
            "type(one_load,(id,load_id)).\n",
            "type(two_load,(id,load_id)).\n",
            "type(three_load,(id,load_id)).\n",
            "type(circle,(id,load_id)).\n",
            "type(triangle,(id,load_id)).\n"
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(has_car(+id,+-car_id)).\n",
            "warmode(has_load(+id,+-car_id,+-load_id)).\n",
            "warmode(short(+id,+-car_id)).\n",
            "warmode(long(+id,+-car_id)).\n",
            "warmode(two_wheels(+id,+-car_id)).\n",
            "warmode(three_wheels(+id,+-car_id)).\n",
            "warmode(roof_open(+id,+-car_id)).\n",
            "warmode(roof_closed(+id,+-car_id)).\n",
            "warmode(zero_load(+id,+-load_id)).\n",
            "warmode(one_load(+id,+-load_id)).\n",
            "warmode(two_load(+id,+-load_id)).\n",
            "warmode(three_load(+id,+-load_id)).\n",
            "warmode(circle(+id,+-load_id)).\n",
            "warmode(triangle(+id,+-load_id)).\n",
            "warmode(edge(+id,+-object,+-object)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,has_car(+puzzle, -car_id)).\n",
            ":- modeb(*,has_car(+puzzle, +car_id)).\n",
            ":- modeb(*,has_load(+puzzle, -car_id, +load_id)).\n",
            ":- modeb(*,has_load(+puzzle, +car_id, -load_id)).\n",
            ":- modeb(*,has_load(+puzzle, -car_id, -load_id)).\n",
            ":- modeb(1,short(+puzzle, -car_id)).\n",
            ":- modeb(1,long(+puzzle, -car_id)).\n",
            ":- modeb(1,two_wheels(+puzzle, -car_id)).\n",
            ":- modeb(1,three_wheels(+puzzle, -car_id)).\n",
            ":- modeb(1,roof_open(+puzzle, -car_id)).\n",
            ":- modeb(1,roof_closed(+puzzle, -car_id)).\n",
            ":- modeb(1,zero_load(+puzzle, -load_id)).\n",
            ":- modeb(1,one_load(+puzzle, -load_id)).\n",
            ":- modeb(1,two_load(+puzzle, -load_id)).\n",
            ":- modeb(1,three_load(+puzzle, -load_id)).\n",
            ":- modeb(1,circle(+puzzle, -load_id)).\n",
            ":- modeb(1,triangle(+puzzle, -load_id)).\n",
            ":- modeb(1,edge(+puzzle, -object, -object)).\n",
            ":- modeb(1,edge(+puzzle, -object, +object)).\n",
            ":- modeb(1,edge(+puzzle, +object, -object)).\n",
            f":- determination({dataset_name}/1,has_car/2).\n",
            f":- determination({dataset_name}/1,has_load/3).\n",
            f":- determination({dataset_name}/1,short/2).\n",
            f":- determination({dataset_name}/1,long/2).\n",
            f":- determination({dataset_name}/1,two_wheels/2).\n",
            f":- determination({dataset_name}/1,three_wheels/2).\n",
            f":- determination({dataset_name}/1,roof_open/2).\n",
            f":- determination({dataset_name}/1,roof_closed/2).\n",
            f":- determination({dataset_name}/1,zero_load/2).\n",
            f":- determination({dataset_name}/1,one_load/2).\n",
            f":- determination({dataset_name}/1,two_load/2).\n",
            f":- determination({dataset_name}/1,three_load).\n"
            f":- determination({dataset_name}/1,circle/2).\n",
            f":- determination({dataset_name}/1,triangle/2).\n",
            f":- determination({dataset_name}/1,edge/3).\n",
        ]
    elif repr == "node_edge":
        popper_bias = [
            "body_pred(has_car,2).\n",
            "body_pred(has_load,3).\n",
            "body_pred(short,2).\n",
            "body_pred(long,2).\n",
            "body_pred(two_wheels,2).\n",
            "body_pred(three_wheels,2).\n",
            "body_pred(roof_open,2).\n",
            "body_pred(roof_closed,2).\n",
            "body_pred(zero_load,2).\n",
            "body_pred(one_load,2).\n",
            "body_pred(two_load,2).\n",
            "body_pred(three_load,2).\n",
            "body_pred(circle,2).\n",
            "body_pred(triangle,2).\n",
            "type(train,(id,)).\n",
            "type(has_car,(id,car_id)).\n",
            "type(has_load,(id,car_id,load_id)).\n",
            "type(short,(id,car_id)).\n",
            "type(long,(id,car_id)).\n",
            "type(two_wheels,(id,car_id)).\n",
            "type(three_wheels,(id,car_id)).\n",
            "type(roof_open,(id,car_id)).\n",
            "type(roof_closed,(id,car_id)).\n",
            "type(zero_load,(id,load_id)).\n",
            "type(one_load,(id,load_id)).\n",
            "type(two_load,(id,load_id)).\n",
            "type(three_load,(id,load_id)).\n",
            "type(circle,(id,load_id)).\n",
            "type(triangle,(id,load_id)).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(has_car(+id,+-car_id)).\n",
            "warmode(has_load(+id,+-car_id,+-load_id)).\n",
            "warmode(short(+id,+-car_id)).\n",
            "warmode(long(+id,+-car_id)).\n",
            "warmode(two_wheels(+id,+-car_id)).\n",
            "warmode(three_wheels(+id,+-car_id)).\n",
            "warmode(roof_open(+id,+-car_id)).\n",
            "warmode(roof_closed(+id,+-car_id)).\n",
            "warmode(zero_load(+id,+-load_id)).\n",
            "warmode(one_load(+id,+-load_id)).\n",
            "warmode(two_load(+id,+-load_id)).\n",
            "warmode(three_load(+id,+-load_id)).\n",
            "warmode(circle(+id,+-load_id)).\n",
            "warmode(triangle(+id,+-load_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,has_car(+puzzle, -car_id)).\n",
            ":- modeb(*,has_car(+puzzle, +car_id)).\n",
            ":- modeb(*,has_load(+puzzle, -car_id, +load_id)).\n",
            ":- modeb(*,has_load(+puzzle, +car_id, -load_id)).\n",
            ":- modeb(*,has_load(+puzzle, -car_id, -load_id)).\n",
            ":- modeb(1,short(+puzzle, -car_id)).\n",
            ":- modeb(1,long(+puzzle, -car_id)).\n",
            ":- modeb(1,two_wheels(+puzzle, -car_id)).\n",
            ":- modeb(1,three_wheels(+puzzle, -car_id)).\n",
            ":- modeb(1,roof_open(+puzzle, -car_id)).\n",
            ":- modeb(1,roof_closed(+puzzle, -car_id)).\n",
            ":- modeb(1,zero_load(+puzzle, -load_id)).\n",
            ":- modeb(1,one_load(+puzzle, -load_id)).\n",
            ":- modeb(1,two_load(+puzzle, -load_id)).\n",
            ":- modeb(1,three_load(+puzzle, -load_id)).\n",
            ":- modeb(1,circle(+puzzle, -load_id)).\n",
            ":- modeb(1,triangle(+puzzle, -load_id)).\n",
            f":- determination({dataset_name}/1,has_car/2).\n",
            f":- determination({dataset_name}/1,has_load/3).\n",
            f":- determination({dataset_name}/1,short/2).\n",
            f":- determination({dataset_name}/1,long/2).\n",
            f":- determination({dataset_name}/1,two_wheels/2).\n",
            f":- determination({dataset_name}/1,three_wheels/2).\n",
            f":- determination({dataset_name}/1,roof_open/2).\n",
            f":- determination({dataset_name}/1,roof_closed/2).\n",
            f":- determination({dataset_name}/1,zero_load/2).\n",
            f":- determination({dataset_name}/1,one_load/2).\n",
            f":- determination({dataset_name}/1,two_load/2).\n",
            f":- determination({dataset_name}/1,three_load).\n"
            f":- determination({dataset_name}/1,circle/2).\n",
            f":- determination({dataset_name}/1,triangle/2).\n",
        ]
    elif repr == "edge_based":
        popper_bias = [
            "body_pred(has_car,2).\n",
            "body_pred(has_load,3).\n",
            "body_pred(edge,3).\n",
            "body_pred(instance,2).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(has_car(+id,+-car_id)).\n",
            "warmode(has_load(+id,+-car_id,+-load_id)).\n",
            "warmode(edge(+id,+-object,+-object)).\n",
            "warmode(instance(+id,+-object)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,has_car(+puzzle, -car_id)).\n",
            ":- modeb(*,has_car(+puzzle, +car_id)).\n",
            ":- modeb(*,has_load(+puzzle, -car_id, +load_id)).\n",
            ":- modeb(*,has_load(+puzzle, +car_id, -load_id)).\n",
            ":- modeb(*,has_load(+puzzle, -car_id, -load_id)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            ":- modeb(*,instance(+puzzle, -object)).\n",
            f":- determination({dataset_name}/1,has_car/2).\n",
            f":- determination({dataset_name}/1,has_load/3).\n",
            f":- determination({dataset_name}/1,edge/3).\n",
            f":- determination({dataset_name}/1,instance/2).\n",
        ]
    elif repr == "Klog":
        popper_bias = [
            "body_pred(has_car,2).\n",
            "body_pred(has_load,2).\n",
            "body_pred(short,2).\n",
            "body_pred(long,2).\n",
            "body_pred(two_wheels,2).\n",
            "body_pred(three_wheels,2).\n",
            "body_pred(roof_open,2).\n",
            "body_pred(roof_closed,2).\n",
            "body_pred(zero_load,2).\n",
            "body_pred(one_load,2).\n",
            "body_pred(two_load,2).\n",
            "body_pred(three_load,2).\n",
            "body_pred(circle,2).\n",
            "body_pred(triangle,2).\n",
            "body_pred(edge,3).\n",
            "type(train,(id,)).\n",
            "type(has_car,(id,car_id)).\n",
            "type(has_load,(id,load_id)).\n",
            "type(short,(id,car_id)).\n",
            "type(long,(id,car_id)).\n",
            "type(two_wheels,(id,car_id)).\n",
            "type(three_wheels,(id,car_id)).\n",
            "type(roof_open,(id,car_id)).\n",
            "type(roof_closed,(id,car_id)).\n",
            "type(zero_load,(id,load_id)).\n",
            "type(one_load,(id,load_id)).\n",
            "type(two_load,(id,load_id)).\n",
            "type(three_load,(id,load_id)).\n",
            "type(circle,(id,load_id)).\n",
            "type(triangle,(id,load_id)).\n"
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(has_car(+id,+-car_id)).\n",
            "warmode(has_load(+id,+-load_id)).\n",
            "warmode(short(+id,+-car_id)).\n",
            "warmode(long(+id,+-car_id)).\n",
            "warmode(two_wheels(+id,+-car_id)).\n",
            "warmode(three_wheels(+id,+-car_id)).\n",
            "warmode(roof_open(+id,+-car_id)).\n",
            "warmode(roof_closed(+id,+-car_id)).\n",
            "warmode(zero_load(+id,+-load_id)).\n",
            "warmode(one_load(+id,+-load_id)).\n",
            "warmode(two_load(+id,+-load_id)).\n",
            "warmode(three_load(+id,+-load_id)).\n",
            "warmode(circle(+id,+-load_id)).\n",
            "warmode(triangle(+id,+-load_id)).\n",
            "warmode(edge(+id,+-object,+-object)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,has_car(+puzzle, -car_id)).\n",
            ":- modeb(*,has_car(+puzzle, +car_id)).\n",
            ":- modeb(*,has_load(+puzzle, +load_id)).\n",
            ":- modeb(*,has_load(+puzzle, -load_id)).\n",
            ":- modeb(1,short(+puzzle, -car_id)).\n",
            ":- modeb(1,long(+puzzle, -car_id)).\n",
            ":- modeb(1,two_wheels(+puzzle, -car_id)).\n",
            ":- modeb(1,three_wheels(+puzzle, -car_id)).\n",
            ":- modeb(1,roof_open(+puzzle, -car_id)).\n",
            ":- modeb(1,roof_closed(+puzzle, -car_id)).\n",
            ":- modeb(1,zero_load(+puzzle, -load_id)).\n",
            ":- modeb(1,one_load(+puzzle, -load_id)).\n",
            ":- modeb(1,two_load(+puzzle, -load_id)).\n",
            ":- modeb(1,three_load(+puzzle, -load_id)).\n",
            ":- modeb(1,circle(+puzzle, -load_id)).\n",
            ":- modeb(1,triangle(+puzzle, -load_id)).\n",
            ":- modeb(1,edge(+puzzle, -object, -object)).\n",
            ":- modeb(1,edge(+puzzle, -object, +object)).\n",
            ":- modeb(1,edge(+puzzle, +object, -object)).\n",
            f":- determination({dataset_name}/1,has_car/2).\n",
            f":- determination({dataset_name}/1,has_load/2).\n",
            f":- determination({dataset_name}/1,short/2).\n",
            f":- determination({dataset_name}/1,long/2).\n",
            f":- determination({dataset_name}/1,two_wheels/2).\n",
            f":- determination({dataset_name}/1,three_wheels/2).\n",
            f":- determination({dataset_name}/1,roof_open/2).\n",
            f":- determination({dataset_name}/1,roof_closed/2).\n",
            f":- determination({dataset_name}/1,zero_load/2).\n",
            f":- determination({dataset_name}/1,one_load/2).\n",
            f":- determination({dataset_name}/1,two_load/2).\n",
            f":- determination({dataset_name}/1,three_load).\n"
            f":- determination({dataset_name}/1,circle/2).\n",
            f":- determination({dataset_name}/1,triangle/2).\n",
            f":- determination({dataset_name}/1,edge/3).\n",
        ]
    
    ilpConverter.logicToPopper(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"), label = dataset_name ,bias_given=popper_bias)
    ilpConverter.logicToTilde(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),givesettings=tilde_settings)
    ilpConverter.logicToAleph(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),label= dataset_name,given_settings=aleph_settings)



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

if ORIGINAL:
    ilpConverter = toILP(relational_path=relational_path, logic_path=os.path.join(base_path, "original", "logic"), dataset_name=dataset_name)
    # convert the training data
    ilpConverter.to_kb(output_path=os.path.join(base_path, "original", "logic", dataset_name + ".kb"), test=False)
    # convert the test data
    ilpConverter.to_kb(output_path=os.path.join(base_path, "original", "logic", dataset_name + "_test.kb"), test=True)

    tilde_bias = [
        f"predict({dataset_name}(+B,-C)).\n",
        "warmode(has_car(+id,+-car_id)).\n",
        "warmode(has_load(+id,+-car_id,+-load_id)).\n",
        "warmode(short(+id,+-car_id)).\n",
        "warmode(long(+id,+-car_id)).\n",
        "warmode(two_wheels(+id,+-car_id)).\n",
        "warmode(three_wheels(+id,+-car_id)).\n",
        "warmode(roof_open(+id,+-car_id)).\n",
        "warmode(roof_closed(+id,+-car_id)).\n",
        "warmode(zero_load(+id,+-load_id)).\n",
        "warmode(one_load(+id,+-load_id)).\n",
        "warmode(two_load(+id,+-load_id)).\n",
        "warmode(three_load(+id,+-load_id)).\n",
        "warmode(circle(+id,+-load_id)).\n",
        "warmode(triangle(+id,+-load_id)).\n",
    ]
    ilpConverter.toTilde(tilde_bias)
    tilde = Tilde(dataset_name=dataset_name, relational_path=relational_path, target=target)
    tilde_results = tilde.run(tilde_input_path=os.path.join(base_path, "original"))
    results = pd.concat([results,tilde_results])

    popper_bias = [
        "head_pred(train,1).\n",
        "body_pred(has_car,2).\n",
        "body_pred(has_load,3).\n",
        "body_pred(short,2).\n",
        "body_pred(long,2).\n",
        "body_pred(two_wheels,2).\n",
        "body_pred(three_wheels,2).\n",
        "body_pred(roof_open,2).\n",
        "body_pred(roof_closed,2).\n",
        "body_pred(zero_load,2).\n",
        "body_pred(one_load,2).\n",
        "body_pred(two_load,2).\n",
        "body_pred(three_load,2).\n",
        "body_pred(circle,2).\n",
        "body_pred(triangle,2).\n"
    ]
    ilpConverter.toPopper(popper_bias)
    popper = Popper(name=dataset_name,relational_path=relational_path,target=target)
    popper_results = popper.run(representation_path=os.path.join(base_path, "original"))
    results = pd.concat([results,popper_results])

    results.to_csv(os.path.join(base_path,"results","results_logic_original.csv"),index=False)
    

# merge the results representations
total_results = pd.DataFrame()
for repr in representations:
    results = pd.read_csv(os.path.join(base_path,"results",f"results_logic_{repr}.csv"))
    os.remove(os.path.join(base_path,"results",f"results_logic_{repr}.csv"))
    total_results = pd.concat([total_results,results])
total_results.to_csv(os.path.join(base_path,"results","results_logic.csv"),index=False)


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
