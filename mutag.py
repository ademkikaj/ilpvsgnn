import os
from sklearn.model_selection import train_test_split
from Benchmark.benchmark import Benchmark
import torch    
import pandas as pd
import yaml

benchmark = Benchmark()

dataset_name = "mutag"
representations = ["node_only","node_edge",'edge_based','Klog']
representations = ["node_only"] 
target = "id"
problem_id = "class"
ilp_systems = ['tilde','aleph','popper']

TILDE = True
ALEPH = False
POPPER = False
GNN = False


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
from Benchmark.mutag.toGraph import toGraph
from Benchmark.mutag.toLogic import toLogic
from Benchmark.toILP import toILP

graph_converter = toGraph(relational_path=relational_path_train,dataset_name=dataset_name,dataset_problem_key="drug_id",target='active')
graph_converter_test = toGraph(relational_path=relational_path_test,dataset_name=dataset_name,dataset_problem_key="drug_id",target='active')

logic_converter = toLogic(dataset_name=dataset_name,relational_path=relational_path_train,problem_key="id")
logic_converter_test = toLogic(dataset_name=dataset_name,relational_path=relational_path_test,problem_key="id")

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
            "body_pred(bond,3).\n",
            "body_pred(atom,3).\n",
            "body_pred(drug,5).\n",
            "body_pred(nitro,2).\n",
            "body_pred(benzene,2).\n",
            "body_pred(edge,3).\n",
            "type(mutag,(id,)).\n",
            "type(bond,(id,node_id,bond_type)).\n",
            "type(atom,(id,node_id,charge)).\n",
            "type(drug,(id,node_id,ind1,inda,logp,lumo)).\n",
            "type(nitro,(id,node_id)).\n",
            "type(benzene,(id,node_id)).\n",
            "type(edge,(id,node_id,node_id)).\n"
        ]
        tilde_settings = [
            "predict(mutag(+B,-C)).\n",
            "warmode(atom(+id,+-node_id,+-charge)).\n",
            "warmode(drug(+id,+-node_id,+-ind1,+-inda,+-logp,+-lumo)).\n",
            "warmode(bond(+id,+-node_id,+-bond_type)).\n",
            "warmode(nitro(+id,+-node_id)).\n",
            "warmode(benzene(+id,+-node_id)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,atom(+puzzle, -node_id, -charge)).\n",
            ":- modeb(*,drug(+puzzle, -node_id, -ind1, -inda, -logp, -lumo)).\n",
            ":- modeb(*,bond(+puzzle, -node_id,-bond_type)).\n",
            ":- modeb(*,nitro(+puzzle, -node_id)).\n",
            ":- modeb(*,benzene(+puzzle, -node_id)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            ":- determination(mutag/1,atom/3).\n",
            ":- determination(mutag/1,drug/6).\n",
            ":- determination(mutag/1,bond/3).\n",
            ":- determination(mutag/1,nitro/2).\n"
            ":- determination(mutag/1,benzene/2).\n"
            ":- determination(mutag/1,edge/3).\n"
        ]
    elif repr == "node_edge":
        popper_bias = [
            "body_pred(bond,4).\n",
            "body_pred(atom,4).\n",
            "body_pred(nitro,3).\n",
            "body_pred(benzene,3).\n",
            "type(mutag,(id,)).\n",
            "type(bond,(id,node_id,node_id,bond_type)).\n",
            "type(atom,(id,node_id,atom_type,charge)).\n",
            "type(nitro,(id,node_id,node_id)).\n",
            "type(benzene,(id,node_id,node_id)).\n",
        ]
        tilde_settings = [
            "predict(mutag(+B,-C)).\n",
            "warmode(atom(+id,+-node_id,+-atom_type,+-charge)).\n",
            "warmode(bond(+id,+-node_id,+-node_id,+-bond_type)).\n",
            "warmode(nitro(+id,+-node_id,+-node_id)).\n",
            "warmode(benzene(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,atom(+puzzle, -object,-atom_type, -charge)).\n",
            ":- modeb(*,bond(+puzzle, -object, -object,+bond_type)).\n",
            ":- modeb(*,bond(+puzzle, -object, +object,+bond_type)).\n",
            ":- modeb(*,bond(+puzzle, +object, -object,+bond_type)).\n",
            ":- modeb(*,nitro(+puzzle, -object, -object)).\n",
            ":- modeb(*,nitro(+puzzle, -object, +object)).\n",
            ":- modeb(*,nitro(+puzzle, +object, -object)).\n",
            ":- modeb(*,benzene(+puzzle, -object, -object)).\n",
            ":- modeb(*,benzene(+puzzle, -object, +object)).\n",
            ":- modeb(*,benzene(+puzzle, +object, -object)).\n",
            ":- determination(mutag/1,atom/4).\n",
            ":- determination(mutag/1,bond/4).\n",
            ":- determination(mutag/1,nitro/3).\n"
            ":- determination(mutag/1,benzene/3).\n"
        ]
    elif repr == "edge_based":
        popper_bias = [
            "body_pred(bond,4).\n",
            "body_pred(instance,3).\n",
            "body_pred(nitro,3).\n",
            "body_pred(benzene,3).\n",
            "type(mutag,(id,)).\n",
            "type(bond,(id,node_id,node_id,bond_type)).\n",
            "type(instance,(id,atom,node_id)).\n",
            "type(nitro,(id,node_id,node_id)).\n",
            "type(benzene,(id,node_id,node_id)).\n",
        ]
        tilde_settings = [
            "predict(mutag(+B,-C)).\n",
            "warmode(instance(+id,+atom,+-atom_type)).\n",
            "warmode(bond(+id,+-node_id,+-node_id,+-bond_type)).\n",
            "warmode(nitro(+id,+-node_id,+-node_id)).\n",
            "warmode(benzene(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,instance(+puzzle,atom,-object)).\n",
            ":- modeb(*,bond(+puzzle, -object, -object,+bond_type)).\n",
            ":- modeb(*,bond(+puzzle, -object, +object,+bond_type)).\n",
            ":- modeb(*,bond(+puzzle, +object, -object,+bond_type)).\n",
            ":- modeb(*,nitro(+puzzle, -object, -object)).\n",
            ":- modeb(*,nitro(+puzzle, -object, +object)).\n",
            ":- modeb(*,nitro(+puzzle, +object, -object)).\n",
            ":- modeb(*,benzene(+puzzle, -object, -object)).\n",
            ":- modeb(*,benzene(+puzzle, -object, +object)).\n",
            ":- modeb(*,benzene(+puzzle, +object, -object)).\n",
            ":- determination(mutag/1,atom/3).\n",
            ":- determination(mutag/1,bond/4).\n",
            ":- determination(mutag/1,nitro/3).\n"
            ":- determination(mutag/1,benzene/3).\n"
        ]
    elif repr == "Klog":
        popper_bias = [
            "body_pred(atom,4).\n",
            "body_pred(bond,2).\n",
            "body_pred(nitro,2).\n",
            "body_pred(benzene,2).\n",
            "body_pred(edge,3).\n",
            "type(mutag,(id,)).\n",
            "type(atom,(id,node_id,atom_type,charge)).\n",
            "type(bond,(id,node_id)).\n",
            "type(nitro,(id,node_id)).\n",
            "type(benzene,(id,node_id)).\n",
            "type(edge,(id,node_id,node_id)).\n"
        ]
        tilde_settings = [
            "predict(mutag(+B,-C)).\n",
            "warmode(atom(+id,+node_id,+-atom_type,+-charge)).\n",
            "warmode(bond(+id,+-node_id)).\n",
            "warmode(nitro(+id,+-node_id)).\n",
            "warmode(benzene(+id,+-node_id)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,atom(+puzzle,+node_id,-atom_type,-charge)).\n",
            ":- modeb(*,bond(+puzzle,-node_id)).\n",
            ":- modeb(*,nitro(+puzzle,-node_id)).\n",
            ":- modeb(*,benzene(+puzzle,-node_id)).\n",
            ":- modeb(*,edge(+puzzle,-node_id,-node_id)).\n",
            ":- modeb(*,edge(+puzzle,-node_id,+node_id)).\n",
            ":- modeb(*,edge(+puzzle,+node_id,-node_id)).\n",
            ":- determination(mutag/1,atom/4).\n",
            ":- determination(mutag/1,bond/2).\n",
            ":- determination(mutag/1,nitro/2).\n"
            ":- determination(mutag/1,benzene/2).\n"
            ":- determination(mutag/1,edge/3).\n"
        ]
    elif repr == "VirtualNode":
        popper_bias = [
            "body_pred(drug,5).\n",
            "body_pred(c,2).\n",
            "body_pred(n,2).\n",
            "body_pred(o,2).\n",
            "body_pred(h,2).\n",
            "body_pred(cl,2).\n",
            "body_pred(f,2).\n",
            "body_pred(br,2).\n",
            "body_pred(i,2).\n",
            "body_pred(first,3).\n",
            "body_pred(second,3).\n",
            "body_pred(third,3).\n",
            "body_pred(fourth,3).\n",
            "body_pred(fifth,3).\n",
            "body_pred(seventh,3).\n",
            "type(mutag,(id,)).\n",
            "type(drug,(id,node_id,ind1,inda,logp,lumo)).\n",
            "type(c,(id,node_id)).\n",
            "type(n,(id,node_id)).\n",
            "type(o,(id,node_id)).\n",
            "type(h,(id,node_id)).\n",
            "type(cl,(id,node_id)).\n",
            "type(f,(id,node_id)).\n",
            "type(br,(id,node_id)).\n",
            "type(i,(id,node_id)).\n",
            "type(first(id,node_id,node_id)).\n",
            "type(second(id,node_id,node_id)).\n",
            "type(third(id,node_id,node_id)).\n",
            "type(fourth(id,node_id,node_id)).\n",
            "type(fifth(id,node_id,node_id)).\n",
            "type(seventh(id,node_id,node_id)).\n",

        ]
        tilde_settings = [
            "predict(mutag(+B,-C)).\n",
            "warmode(drug(+id,+-ind1,+-inda,+-logp,+-lumo)).\n",
            "warmode(c(+id,+-node_id)).\n",
            "warmode(n(+id,+-node_id)).\n",
            "warmode(o(+id,+-node_id)).\n",
            "warmode(h(+id,+-node_id)).\n",
            "warmode(cl(+id,+-node_id)).\n",
            "warmode(f(+id,+-node_id)).\n",
            "warmode(br(+id,+-node_id)).\n",
            "warmode(i(+id,+-node_id)).\n",
            "warmode(first(+id,+-node_id,+-node_id)).\n",
            "warmode(second(+id,+-node_id,+-node_id)).\n",
            "warmode(third(+id,+-node_id,+-node_id)).\n",
            "warmode(fourth(+id,+-node_id,+-node_id)).\n",
            "warmode(fifth(+id,+-node_id,+-node_id)).\n",
            "warmode(seventh(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            
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
if TILDE or ALEPH or POPPER:
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


