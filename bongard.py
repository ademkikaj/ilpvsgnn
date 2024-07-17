import os
from sklearn.model_selection import train_test_split
from Benchmark.benchmark import Benchmark
import torch
import pandas as pd
import yaml

benchmark = Benchmark()

dataset_name = "bongard"
representations = ["node_only","edge_based","node_edge","Klog"]
target = "class"
problem_id = "problemId"
ilp_systems = ['tilde','aleph','popper']


TILDE = True
ALEPH = False
POPPER = False
GNN = True

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


## split the data into train and test

benchmark.split_relational_data(relational_path, dataset_name,class_name=target,problem_id=problem_id)


# conversions to different representations

import sys
from Benchmark.bongard.toGraph import toGraph
from Benchmark.bongard.toLogic import toLogic
from Benchmark.toILP import toILP

graph_converter = toGraph(relational_path=relational_path_train)
graph_converter_test = toGraph(relational_path=relational_path_test)

logic_converter = toLogic()


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
            "body_pred(square,2).\n",
            "body_pred(circle,2).\n",
            "body_pred(triangle,2).\n",
            "body_pred(in,2).\n",
            "body_pred(edge,3).\n",
            "type(bongard,(id,)).\n",
            "type(square,(id,object)).\n",
            "type(circle,(id,object)).\n",
            "type(triangle,(id,object)).\n",
            "type(in,(id,object)).\n",
            "type(edge,(id,object,object)).\n",
        ]
        tilde_settings = [
            "predict(bongard(+B,-C)).\n",
            "rmode(5,triangle(+id,+-object)).\n",
            "rmode(5,square(+id,+-object)).\n",
            "rmode(5,circle(+id,+-object)).\n",
            "rmode(5,in(+id,+-object)).\n",
            "rmode(5,edge(+id,+-object,+-object)).\n",
            "typed_language(yes).\n",
            "type(bongard(pic,class)).\n",
            "type(triangle(pic,obj)).\n",
            "type(square(pic,obj)).\n",
            "type(circle(pic,obj)).\n",
            "type(in(pic,obj)).\n",
            "type(edge(pic,obj,obj)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,square(+puzzle, -object)).\n",
            ":- modeb(*,circle(+puzzle, -object)).\n",
            ":- modeb(*,triangle(+puzzle, -object)).\n",
            ":- modeb(*,in(+puzzle, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            ":- determination(bongard/1,square/2).\n",
            ":- determination(bongard/1,circle/2).\n",
            ":- determination(bongard/1,triangle/2).\n",
            ":- determination(bongard/1,in/2).\n"
            ":- determination(bongard/1,edge/3).\n"
        ]
    elif repr == "node_edge":
        popper_bias = [
            "body_pred(square,2).\n",
            "body_pred(circle,2).\n",
            "body_pred(triangle,2).\n",
            "body_pred(edge,3).\n",
            "type(bongard,(id,)).\n",
            "type(square,(id,object)).\n",
            "type(circle,(id,object)).\n",
            "type(triangle,(id,object)).\n",
            "type(edge,(id,object,object)).\n",
        ]
        tilde_settings = [
            "predict(bongard(+B,-C)).\n",
            "rmode(5: triangle(+P,+-S)).\n",
            "rmode(5: square(+P,+-S)).\n",
            "rmode(5: circle(+P,+-S)).\n",
            "rmode(5: edge(+P,+S1,+-S2)).\n",
            "rmode(5: instance(+P,+-S)).\n",
            "typed_language(yes).\n",
            "type(bongard(pic,class)).\n",
            "type(triangle(pic,obj)).\n",
            "type(square(pic,obj)).\n",
            "type(circle(pic,obj)).\n",
            "type(edge(pic,obj,obj)).\n",
            "type(instance(pic,obj)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,square(+puzzle, -object)).\n",
            ":- modeb(*,circle(+puzzle, -object)).\n",
            ":- modeb(*,triangle(+puzzle, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            ":- determination(bongard/1,square/2).\n",
            ":- determination(bongard/1,circle/2).\n",
            ":- determination(bongard/1,triangle/2).\n",
            ":- determination(bongard/1,edge/3).\n"
        ]
    elif repr == "edge_based":
        popper_bias = [
            "body_pred(shape,3).\n",
            "body_pred(in,3).\n",
            "type(bongard,(id,)).\n",
            "type(in,(id,object,object)).\n"
            "type(shape,(id,fact,object)).\n"
        ]
        tilde_settings = [
            "predict(bongard(+B,-C)).\n",
            "rmode(5: shape(+P,triangle,+-C)).\n",
            "rmode(5: shape(+P,square,+-C)).\n",
            "rmode(5: shape(+P,circle,+-C)).\n",
            "rmode(5: in(+P,+S1,+-S2)).\n",
            "typed_language(yes).\n",
            "type(bongard(pic,class)).\n",
            "type(in(pic,obj,obj)).\n",
            "type(shape(pic,fact,obj)).\n",
        ]
        aleph_settings = [
            ":- modeb(1,square(+constant)).\n",
            ":- modeb(1,circle(+constant)).\n",
            ":- modeb(1,triangle(+constant)).\n",
            ":- modeb(*,shape(+puzzle, +-constant, -object)).\n",
            ":- modeb(*,in(+puzzle, -object, -object)).\n",
            ":- modeb(*,in(+puzzle, -object, +object)).\n",
            ":- modeb(*,in(+puzzle, +object, -object)).\n",
            ":- determination(bongard/1,shape/3).\n",
            ":- determination(bongard/1,in/3).\n"
        ]
    elif repr == "Klog":
        popper_bias = [
            "body_pred(square,2).\n",
            "body_pred(circle,2).\n",
            "body_pred(triangle,2).\n",
            "body_pred(edge,3).\n",
            "body_pred(in,2).\n",
            "type(bongard,(id,)).\n",
            "type(square,(id,object)).\n",
            "type(circle,(id,object)).\n",
            "type(triangle,(id,object)).\n",
            "type(in,(id,object)).\n",
            "type(edge,(id,object,object)).\n",
        ]
        tilde_settings = [
            "predict(bongard(+B,-C)).\n",
            "warmode(triangle(+id,+-object)).\n",
            "warmode(square(+id,+-object)).\n",
            "warmode(circle(+id,+-object)).\n",
            "warmode(edge(+id,+-object,+-object)).\n",
            "warmode(in(+id,+-object)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,square(+puzzle, +-object)).\n",
            ":- modeb(*,circle(+puzzle, +-object)).\n",
            ":- modeb(*,triangle(+puzzle, +-object)).\n",
            ":- modeb(*,in(+puzzle, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            ":- determination(bongard/1,square/2).\n",
            ":- determination(bongard/1,circle/2).\n",
            ":- determination(bongard/1,triangle/2).\n",
            ":- determination(bongard/1,in/2).\n"
            ":- determination(bongard/1,edge/3).\n"
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

