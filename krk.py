import os
from sklearn.model_selection import train_test_split
from Benchmark.benchmark import Benchmark
import torch
import pandas as pd
import yaml

benchmark = Benchmark()

dataset_name = "krk"
representations = ["node_only","node_edge",'edge_based','Klog']
representations = ["node_edge"]
target = "class"
problem_id = "id"
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
from Benchmark.KRK.toGraph import toGraph
from Benchmark.KRK.toLogic import toLogic
from Benchmark.toILP import toILP

graph_converter = toGraph(relational_path=relational_path_train,
                          dataset_name=dataset_name,
                          dataset_problem_key=problem_id,
                          target=target)
graph_converter_test = toGraph(relational_path=relational_path_test,dataset_name=dataset_name,dataset_problem_key=problem_id,target=target)

logic_converter = toLogic(dataset_name=dataset_name)


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
if TILDE or POPPER or ALEPH: 
    for repr in representations:

        ilpConverter = toILP(relational_path=relational_path, logic_path=os.path.join(base_path, repr, "logic"), dataset_name=dataset_name)
        
        # Popper 
        if repr == "node_only":
            popper_bias = [
                "body_pred(white_king,4).\n",
                "body_pred(white_rook,4).\n",
                "body_pred(black_king,4).\n",
                "body_pred(edge,3).\n",
                f"type({dataset_name},(id,)).\n",
                "type(white_king,(id,node_id,x,y)).\n",
                "type(white_rook,(id,node_id,x,y)).\n",
                "type(black_king,(id,node_id,x,y)).\n",
                "type(edge,(id,node_id,node_id)).\n"
            ]
            tilde_settings = [
                f"predict({dataset_name}(+B,-C)).\n",
                "warmode(white_king(+id,+-node_id,+-x,+-y)).\n",
                "warmode(white_rook(+id,+-node_id,+-x,+-y)).\n",
                "warmode(black_king(+id,+-node_id,+-x,+-y)).\n",
                "warmode(edge(+id,+-node_id,+-node_id)).\n",
            ]
            aleph_settings = [
                ":- modeb(*,white_king(+puzzle, -node_id, -x, -y)).\n",
                ":- modeb(*,white_rook(+puzzle, -node_id, -x, -y)).\n",
                ":- modeb(*,black_king(+puzzle, -node_id, -x, -y)).\n",
                ":- modeb(*,edge(+puzzle, -node_id, -node_id)).\n",
                ":- modeb(*,edge(+puzzle, -node_id, +node_id)).\n",
                ":- modeb(*,edge(+puzzle, +node_id, -node_id)).\n",            
                f":- determination({dataset_name}/1,white_king/4).\n",
                f":- determination({dataset_name}/1,white_rook/4).\n",
                f":- determination({dataset_name}/1,black_king/4).\n",
                f":- determination({dataset_name}/1,edge/3).\n"
            ]
        elif repr == "node_edge":
            popper_bias = [
                "body_pred(white_king,4).\n",
                "body_pred(white_rook,4).\n",
                "body_pred(black_king,4).\n",
                "body_pred(edge,3).\n",
                "body_pred(same_rank,3).\n",
                "body_pred(same_file,3).\n",
                f"type({dataset_name},(id,)).\n",
                "type(white_king,(id,node_id,x,y)).\n",
                "type(white_rook,(id,node_id,x,y)).\n",
                "type(black_king,(id,node_id,x,y)).\n",
                "type(edge,(id,node_id,node_id)).\n",
                "type(same_rank,(id,node_id,node_id)).\n",
                "type(same_file,(id,node_id,node_id)).\n",
            ]
            tilde_settings = [
                f"predict({dataset_name}(+B,-C)).\n",
                "warmode(white_king(+id,+node_id,+-x,+-y)).\n",
                "warmode(white_rook(+id,+node_id,+-x,+-y)).\n",
                "warmode(black_king(+id,+node_id,+-x,+-y)).\n",
                "warmode(edge(+id,+-node_id,+-node_id)).\n",
                "warmode(same_rank(+id,+-node_id,+-node_id)).\n",
                "warmode(same_file(+id,+-node_id,+-node_id)).\n", 
            ]
            aleph_settings = [
                ":- modeb(*,white_king(+puzzle, -node_id, #int,#int)).\n",
                ":- modeb(*,white_king(+puzzle, +node_id, #int,#int)).\n",
                ":- modeb(*,white_rook(+puzzle, -node_id, #int,#int)).\n",
                ":- modeb(*,white_rook(+puzzle, +node_id, #int,#int)).\n",
                ":- modeb(*,black_king(+puzzle, -node_id, #int,#int)).\n",
                ":- modeb(*,black_king(+puzzle, +node_id, #int,#int)).\n",
                ":- modeb(*,edge(+puzzle, -node_id, -node_id)).\n",
                ":- modeb(*,edge(+puzzle, -node_id, +node_id)).\n",
                ":- modeb(*,edge(+puzzle, +node_id, -node_id)).\n",
                ":- modeb(*,same_rank(+puzzle, -node_id, -node_id)).\n",
                ":- modeb(*,same_rank(+puzzle, -node_id, +node_id)).\n",
                ":- modeb(*,same_rank(+puzzle, +node_id, -node_id)).\n",
                ":- modeb(*,same_file(+puzzle, -node_id, -node_id)).\n",
                ":- modeb(*,same_file(+puzzle, -node_id, +node_id)).\n",  
                ":- modeb(*,same_file(+puzzle, +node_id, -node_id)).\n", 
                f":- determination({dataset_name}/1,white_king/4).\n",
                f":- determination({dataset_name}/1,white_rook/4).\n",
                f":- determination({dataset_name}/1,black_king/4).\n",
                f":- determination({dataset_name}/1,edge/3).\n",
                f":- determination({dataset_name}/1,same_rank/3).\n",
                f":- determination({dataset_name}/1,same_file/3).\n",
            ]
        elif repr == "edge_based":
            popper_bias = [
                "body_pred(white_king,4).\n",
                "body_pred(white_rook,4).\n",
                "body_pred(black_king,4).\n",
                "body_pred(edge,3).\n",
                "body_pred(same_rank,3).\n",
                "body_pred(same_file,3).\n",
                "body_pred(instance,3).\n",
                f"type({dataset_name},(id,)).\n",
                "type(white_king,(id,node_id,x,y)).\n",
                "type(white_rook,(id,node_id,x,y)).\n",
                "type(black_king,(id,node_id,x,y)).\n",
                "type(edge,(id,node_id,node_id)).\n",
                "type(same_rank,(id,node_id,node_id)).\n",
                "type(same_file,(id,node_id,node_id)).\n",
                "type(instance,(id,white_king,node_id)).\n",
                "type(instance,(id,white_rook,node_id)).\n",
                "type(instance,(id,black_king,node_id)).\n",
            ]
            tilde_settings = [
                f"predict({dataset_name}(+B,-C)).\n",
                "warmode(white_king(+id,+node_id,+-x,+-y)).\n",
                "warmode(white_rook(+id,+node_id,+-x,+-y)).\n",
                "warmode(black_king(+id,+node_id,+-x,+-y)).\n",
                "warmode(edge(+id,+-node_id,+-node_id)).\n",
                "warmode(same_rank(+id,+-node_id,+-node_id)).\n",
                "warmode(same_file(+id,+-node_id,+-node_id)).\n",
                "warmode(instance(+id,white_king,+-node_id)).\n",
                "warmode(instance(+id,white_rook,+-node_id)).\n",
                "warmode(instance(+id,black_king,+-node_id)).\n",
            ]
            aleph_settings = [
                ":- modeb(*,white_king(+puzzle, -object, -x, -y)).\n",
                ":- modeb(*,white_king(+puzzle, +object, -x, -y)).\n",
                ":- modeb(*,white_rook(+puzzle, -object, -x, -y)).\n",
                ":- modeb(*,white_rook(+puzzle, +object, -x, -y)).\n",
                ":- modeb(*,black_king(+puzzle, -object, -x, -y)).\n",
                ":- modeb(*,black_king(+puzzle, +object, -x, -y)).\n",
                ":- modeb(*,edge(+puzzle, -object, -object)).\n",
                ":- modeb(*,edge(+puzzle, -object, +object)).\n",
                ":- modeb(*,edge(+puzzle, +object, -object)).\n",
                ":- modeb(*,same_rank(+puzzle, -object, -object)).\n",
                ":- modeb(*,same_rank(+puzzle, -object, +object)).\n",
                ":- modeb(*,same_rank(+puzzle, +object, -object)).\n",
                ":- modeb(*,same_file(+puzzle, -object, -object)).\n",
                ":- modeb(*,same_file(+puzzle, -object, +object)).\n",
                ":- modeb(*,same_file(+puzzle, +object, -object)).\n",
                ":- modeb(*,instance(+puzzle, white_king, -object)).\n", 
                ":- modeb(*,instance(+puzzle, white_rook, -object)).\n",
                ":- modeb(*,instance(+puzzle, black_king, -object)).\n",        
                f":- determination({dataset_name}/1,white_king/4).\n",
                f":- determination({dataset_name}/1,white_rook/4).\n",
                f":- determination({dataset_name}/1,black_king/4).\n",
                f":- determination({dataset_name}/1,edge/3).\n",
                f":- determination({dataset_name}/1,same_rank/3).\n",
                f":- determination({dataset_name}/1,same_file/3).\n",
                f":- determination({dataset_name}/1,instance/3).\n"
            ]
        elif repr == "Klog":
            popper_bias = [
                "body_pred(white_king,4).\n",
                "body_pred(white_rook,4).\n",
                "body_pred(black_king,4).\n",
                "body_pred(edge,3).\n",
                "body_pred(same_rank,2).\n",
                "body_pred(same_file,2).\n",
                f"type({dataset_name},(id,)).\n",
                "type(white_king,(id,node_id,x,y)).\n",
                "type(white_rook,(id,node_id,x,y)).\n",
                "type(black_king,(id,node_id,x,y)).\n",
                "type(edge,(id,node_id,node_id)).\n",
                "type(same_rank,(id,node_id)).\n",
                "type(same_file,(id,node_id)).\n",
            ]
            tilde_settings = [
                "predict(krk(+B,-C)).\n",
                "warmode(white_king(+id,+node_id,+-x,+-y)).\n",
                "warmode(white_rook(+id,+node_id,+-x,+-y)).\n",
                "warmode(black_king(+id,+node_id,+-x,+-y)).\n",
                "warmode(same_rank(+id,+-node_id)).\n",
                "warmode(same_file(+id,+-node_id)).\n",
                "warmode(edge(+id,+-node_id,+-node_id)).\n",
            ]
            aleph_settings = [
                ":- modeb(*,white_king(+puzzle, -object, -x, -y)).\n",
                ":- modeb(*,white_king(+puzzle, +object, -x, -y)).\n",
                ":- modeb(*,white_rook(+puzzle, -object, -x, -y)).\n",
                ":- modeb(*,white_rook(+puzzle, +object, -x, -y)).\n",
                ":- modeb(*,black_king(+puzzle, -object, -x, -y)).\n",
                ":- modeb(*,black_king(+puzzle, +object, -x, -y)).\n",
                ":- modeb(*,edge(+puzzle, -object, -object)).\n",
                ":- modeb(*,edge(+puzzle, -object, +object)).\n",
                ":- modeb(*,edge(+puzzle, +object, -object)).\n",
                ":- modeb(*,same_rank(+puzzle, -object)).\n",
                ":- modeb(*,same_file(+puzzle, -object)).\n",     
                f":- determination({dataset_name}/1,white_king/4).\n",
                f":- determination({dataset_name}/1,white_rook/4).\n",
                f":- determination({dataset_name}/1,black_king/4).\n",
                f":- determination({dataset_name}/1,edge/3).\n",
                f":- determination({dataset_name}/1,same_rank/2).\n",
                f":- determination({dataset_name}/1,same_file/2).\n",
            ]
        elif repr == "FullBoard":
            popper_bias = [
                "body_pred(white_king,2).\n",
                "body_pred(white_rook,2).\n",
                "body_pred(black_king,2).\n",
                "body_pred(edge,3).\n",
                "type(krk,(id,class)).\n",
                "type(white_king,(id,node_id)).\n",
                "type(white_rook,(id,node_id)).\n",
                "type(black_king,(id,node_id)).\n",
                "type(edge,(id,node_id,node_id)).\n",
            ]
            tilde_settings = [
                "predict(krk(+B,-C)).\n",
                "warmode(white_king(+id,+-node_id)).\n",
                "warmode(white_rook(+id,+-node_id)).\n",
                "warmode(black_king(+id,+-node_id)).\n",
                "warmode(edge(+id,+-node_id,+-node_id)).\n",
            ]
            aleph_settings = [
                ":- modeb(*,white_king(+puzzle, -node_id)).\n",
                ":- modeb(*,white_rook(+puzzle, -node_id)).\n",
                ":- modeb(*,black_king(+puzzle, -node_id)).\n",
                ":- modeb(*,edge(+puzzle, -object, -object)).\n",
                ":- modeb(*,edge(+puzzle, -object, +object)).\n",
                ":- modeb(*,edge(+puzzle, +object, -object)).\n",            
                f":- determination({dataset_name}/1,white_king/2).\n",
                f":- determination({dataset_name}/1,white_rook/2).\n",
                f":- determination({dataset_name}/1,black_king/2).\n",
                f":- determination({dataset_name}/1,edge/3).\n",
            ]
        elif repr == "FullDiag":
            popper_bias = [
                "body_pred(white_king,2).\n",
                "body_pred(white_rook,2).\n",
                "body_pred(black_king,2).\n",
                "body_pred(edge,3).\n",
                "type(krk,(id,class)).\n",
                "type(white_king,(id,node_id)).\n",
                "type(white_rook,(id,node_id)).\n",
                "type(black_king,(id,node_id)).\n",
                "type(edge,(id,node_id,node_id)).\n",
            ]
            tilde_settings = [
                "predict(krk(+B,-C)).\n",
                "warmode(white_king(+id,+-node_id)).\n",
                "warmode(white_rook(+id,+-node_id)).\n",
                "warmode(black_king(+id,+-node_id)).\n",
                "warmode(edge(+id,+-node_id,+-node_id)).\n",
            ]
            aleph_settings = [
                ":- modeb(*,white_king(+puzzle, -node_id)).\n",
                ":- modeb(*,white_rook(+puzzle, -node_id)).\n",
                ":- modeb(*,black_king(+puzzle, -node_id)).\n",
                ":- modeb(*,edge(+puzzle, -object, -object)).\n",
                ":- modeb(*,edge(+puzzle, -object, +object)).\n",
                ":- modeb(*,edge(+puzzle, +object, -object)).\n",            
                f":- determination({dataset_name}/1,white_king/2).\n",
                f":- determination({dataset_name}/1,white_rook/2).\n",
                f":- determination({dataset_name}/1,black_king/2).\n",
                f":- determination({dataset_name}/1,edge/3).\n",
            ]

        ilpConverter.logicToPopper(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"), label = dataset_name ,bias_given=popper_bias)
        ilpConverter.logicToTilde(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),givesettings=tilde_settings)
        ilpConverter.logicToAleph(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),label= dataset_name,given_settings=aleph_settings)

        # append background prolog function to aleph file
        with open(os.path.join(base_path, repr, "logic", "aleph", "background.pl"), "a") as file:
            file.write(f"same_row(A,B) :- white_rook(_,_,A,_),white_king(_,_,B,_),A=:=B.\n")
            file.write(f"same_col(A,B) :- white_rook(_,_,_,A),black_king(_,_,_,B),A=:=B.\n")
        

        # # append background prolog function to the tilde file
        # with open(os.path.join(base_path, repr, "logic", "tilde", "krk.bg"), "a") as file:
        #     file.write(f"same_row(A,B) :- white_rook(_,_,A,_),white_king(_,_,B,_),A=:=B.\n")
        #     file.write(f"same_col(A,B) :- white_rook(_,_,_,A),black_king(_,_,_,B),A=:=B.\n")
        
        # # append background prolog function to the popper file
        # with open(os.path.join(base_path, repr, "logic", "popper", "bk.pl"), "a") as file:
        #     file.write(f"same_row(A,B) :- white_rook(_,_,A,_),white_king(_,_,B,_),A=:=B.\n")
        #     file.write(f"same_col(A,B) :- white_rook(_,_,_,A),black_king(_,_,_,B),A=:=B.\n")
        
        

## open the test data and turn the target column into pos and neg
test_data = pd.read_csv(os.path.join(relational_path_test, f"{dataset_name}.csv"))
test_data[target] = test_data[target].apply(lambda x: "pos" if x == "legal" else "neg")
test_data.to_csv(os.path.join(relational_path_test, f"{dataset_name}.csv"), index=False)



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
if TILDE or POPPER or ALEPH:
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

