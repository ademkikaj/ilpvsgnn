import os
from sklearn.model_selection import train_test_split
from Benchmark.benchmark import Benchmark
import torch
import pandas as pd
import yaml

benchmark = Benchmark()

dataset_name = "cora"
representations = ["node_only","node_edge",'edge_based','Klog']
representations = ["node_only","node_edge","Klog"]
representations = []
target = "class_label"
problem_id = "paper_id"
ilp_systems = ['tilde','aleph','popper']

ORIGINAL = True

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# print current directory

base_path = os.path.join("docker", "Benchmark", dataset_name)

# prepare the csv files
benchmark.prepare_csv(os.path.join("docker","Benchmark",dataset_name,"relational"))

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


# conversions to different representations

import sys
from Benchmark.University.toGraph import toGraph
from Benchmark.University.toLogic import toLogic
from Benchmark.toILP import toILP

graph_converter = toGraph(relational_path=relational_path_train,dataset_name=dataset_name,dataset_problem_key=problem_id,target=target)
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
    
for repr in representations:

    ilpConverter = toILP(relational_path=relational_path, logic_path=os.path.join(base_path, repr, "logic"), dataset_name=dataset_name)
    
    # Popper 
    if repr == "node_only":
        popper_bias = [
            "body_pred(students,3).\n",
            "body_pred(course,4).\n",
            "body_pred(professor,4).\n",
            "body_pred(registered,4).\n",
            "body_pred(ra,4).\n",
            "body_pred(edge,3).\n",
            f"type({dataset_name},(id,)).\n",
            "type(students,(id,node_id,ranking)).\n",
            "type(course,(id,course_id,diff,rating)).\n",
            "type(professor,(id,prof_id,teaching,pop)).\n",
            "type(registered,(id,course_id,grade,satis)).\n",
            "type(ra,(id,prof_id,sal,cap)).\n",
            "type(edge,(id,node_id,node_id)).\n"
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(students(+id,+node_id,+-ranking)).\n",
            "warmode(course(+id,+-course_id,+-diff,+-rating)).\n",
            "warmode(professor(+id,+-prof_id,+-teaching,+-pop)).\n",
            "warmode(registered(+id,+-course_id,+-grade,+-satis)).\n",
            "warmode(ra(+id,+-prof_id,+-sal,+-cap)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,students(+puzzle, -node_id, -ranking)).\n",
            ":- modeb(*,course(+puzzle, -course_id, -diff, -rating)).\n",
            ":- modeb(*,professor(+puzzle, -prof_id, -teaching, -pop)).\n",
            ":- modeb(*,registered(+puzzle, -course_id, -grade, -satis)).\n",
            ":- modeb(*,ra(+puzzle, -prof_id, -sal, -cap)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            f":- determination({dataset_name}/1,students/3).\n",
            f":- determination({dataset_name}/1,course/4).\n",
            f":- determination({dataset_name}/1,professor/4).\n",
            f":- determination({dataset_name}/1,registered/4).\n",
            f":- determination({dataset_name}/1,ra/4).\n",
            f":- determination({dataset_name}/1,edge/3).\n",
        ]
    elif repr == "node_edge":
        popper_bias = [
            "body_pred(students,3).\n",
            "body_pred(course,4).\n",
            "body_pred(professor,4).\n",
            "body_pred(registered,4).\n",
            "body_pred(ra,4).\n",
            f"type({dataset_name},(id,)).\n",
            "type(students,(id,node_id,ranking)).\n",
            "type(course,(id,course_id,diff,rating)).\n",
            "type(professor,(id,prof_id,teaching,pop)).\n",
            "type(registered,(id,course_id,grade,satis)).\n",
            "type(ra,(id,prof_id,sal,cap)).\n",
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(students(+id,+-node_id,+-ranking)).\n",
            "warmode(course(+id,+-course_id,+-diff,+-rating)).\n",
            "warmode(professor(+id,+-prof_id,+-teaching,+-pop)).\n",
            "warmode(registered(+id,+-course_id,+-grade,+-satis)).\n",
            "warmode(ra(+id,+-prof_id,+-sal,+-cap)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,students(+puzzle, -node_id, -ranking)).\n",
            ":- modeb(*,course(+puzzle, -course_id, -diff, -rating)).\n",
            ":- modeb(*,professor(+puzzle, -prof_id, -teaching, -pop)).\n",
            ":- modeb(*,registered(+puzzle, -course_id, -grade, -satis)).\n",
            ":- modeb(*,ra(+puzzle, -prof_id, -sal, -cap)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            f":- determination({dataset_name}/1,students/3).\n",
            f":- determination({dataset_name}/1,course/4).\n",
            f":- determination({dataset_name}/1,professor/4).\n",
            f":- determination({dataset_name}/1,registered/4).\n",
            f":- determination({dataset_name}/1,RA/4).\n",
            f":- determination({dataset_name}/1,edge/3).\n",
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
            "rmode(5: white_king(+P,+-S,+-X,+-Y)).\n",
            "rmode(5: white_rook(+P,+-S,+-X,+-Y)).\n",
            "rmode(5: black_king(+P,+-S,+-X,+-Y)).\n",
            "rmode(5: edge(+P,+S1,+-S2)).\n",
            "rmode(5: same_rank(+P,+S1,+-S2)).\n",
            "rmode(5: same_file(+P,+S1,+-S2)).\n",
            "rmode(5: instance(+P,white_king,+-S2)).\n",
            "rmode(5: instance(+P,white_rook,+-S2)).\n",
            "rmode(5: instance(+P,black_king,+-S2)).\n",
            "typed_language(yes).\n",
            f"type({dataset_name}(id,class)).\n",
            "type(white_king(id,node_id,x,y)).\n",
            "type(white_rook(id,node_id,x,y)).\n",
            "type(black_king(id,node_id,x,y)).\n",
            "type(edge(id,node_id,node_id)).\n"
            "type(same_rank(id,node_id,node_id)).\n",
            "type(same_file(id,node_id,node_id)).\n"
            "type(instance(id,white_king,node_id)).\n"
            "type(instance(id,white_rook,node_id)).\n"
            "type(instance(id,black_king,node_id)).\n"
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
            ":- modeb(*,white_king(+puzzle, -node_id, -x, -y)).\n",
            ":- modeb(*,white_rook(+puzzle, -node_id, -x, -y)).\n",
            ":- modeb(*,black_king(+puzzle, -node_id, -x, -y)).\n",
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
            "body_pred(students,3).\n",
            "body_pred(course,4).\n",
            "body_pred(professor,4).\n",
            "body_pred(registered,4).\n",
            "body_pred(ra,4).\n",
            "body_pred(edge,3).\n",
            f"type({dataset_name},(id,)).\n",
            "type(students,(id,node_id,ranking)).\n",
            "type(course,(id,course_id,diff,rating)).\n",
            "type(professor,(id,prof_id,teaching,pop)).\n",
            "type(registered,(id,course_id,grade,satis)).\n",
            "type(ra,(id,prof_id,sal,cap)).\n",
            "type(edge,(id,node_id,node_id)).\n"
        ]
        tilde_settings = [
            f"predict({dataset_name}(+B,-C)).\n",
            "warmode(students(+id,+node_id,+-ranking)).\n",
            "warmode(course(+id,+-course_id,+-diff,+-rating)).\n",
            "warmode(professor(+id,+-prof_id,+-teaching,+-pop)).\n",
            "warmode(registered(+id,+-course_id,+-grade,+-satis)).\n",
            "warmode(ra(+id,+-prof_id,+-sal,+-cap)).\n",
            "warmode(edge(+id,+-node_id,+-node_id)).\n",
        ]
        aleph_settings = [
            ":- modeb(*,students(+puzzle, -node_id, -ranking)).\n",
            ":- modeb(*,course(+puzzle, -course_id, -diff, -rating)).\n",
            ":- modeb(*,professor(+puzzle, -prof_id, -teaching, -pop)).\n",
            ":- modeb(*,registered(+puzzle, -course_id, -grade, -satis)).\n",
            ":- modeb(*,ra(+puzzle, -prof_id, -sal, -cap)).\n",
            ":- modeb(*,edge(+puzzle, -object, -object)).\n",
            ":- modeb(*,edge(+puzzle, -object, +object)).\n",
            ":- modeb(*,edge(+puzzle, +object, -object)).\n",
            f":- determination({dataset_name}/1,students/3).\n",
            f":- determination({dataset_name}/1,course/4).\n",
            f":- determination({dataset_name}/1,professor/4).\n",
            f":- determination({dataset_name}/1,registered/4).\n",
            f":- determination({dataset_name}/1,ra/4).\n",
            f":- determination({dataset_name}/1,edge/3).\n",
        ]
    
    ilpConverter.logicToPopper(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"), label = dataset_name ,bias_given=popper_bias)
    ilpConverter.logicToTilde(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),givesettings=tilde_settings)
    ilpConverter.logicToAleph(logic_file_path=os.path.join(base_path, repr, "logic", dataset_name + ".kb"),label= dataset_name,given_settings=aleph_settings)


from Benchmark.tilde import Tilde
from Benchmark.popper_system import Popper
from Benchmark.aleph_system import Aleph

if ORIGINAL:
    ilpConverter = toILP(relational_path=relational_path, logic_path=os.path.join(base_path, "original", "logic"), dataset_name=dataset_name)
    # convert the training data
    ilpConverter.to_kb(os.path.join(base_path, "original", "logic", f"{dataset_name}.kb"),test=False)
    # convert the test data
    ilpConverter.to_kb(os.path.join(base_path, "original", "logic", f"{dataset_name}_test.kb"),test=True)

    
    tilde_bias = [
        "predict(cora(+B,-C)).\n",
        "warmode(content(+paper_id,+-word_cited_id)).\n",
        "warmode(cites(+-paper_id,+-paper_id)).\n",
    ]
    ilpConverter.toTilde(tilde_bias)
    tilde = Tilde(dataset_name=dataset_name, relational_path=relational_path, target=target)
    tilde_results = tilde.run(tilde_input_path=os.path.join(base_path, "original"))
    print(tilde_results)

# open the train data turn the target column into pos and neg

# All the files are in the correct locations for the logic and graph representations
# Run the ILP systems on the different representations

for repr in representations:
    print("Representation: ",repr)


    tilde = Tilde(dataset_name=dataset_name, relational_path=relational_path, target=target)
    tilde_results = tilde.run(tilde_input_path=os.path.join(base_path,repr))
    tilde_results['representation'] = repr
    print(tilde_results)

    popper = Popper(name=dataset_name,relational_path=relational_path,target=target)
    popper_results = popper.run(representation_path=os.path.join(base_path,repr))
    popper_results['representation'] = repr

    aleph = Aleph(name=dataset_name, relational_path=relational_path,target=target)
    aleph_results = aleph.run(representation_path=os.path.join(base_path,repr))
    aleph_results['representation'] = repr

    results = pd.concat([tilde_results, popper_results, aleph_results])
    results.to_csv(os.path.join(base_path,"results",f"results_logic_{repr}.csv"),index=False)


# merge the results representations
total_results = pd.DataFrame()
for repr in representations:
    results = pd.read_csv(os.path.join(base_path,"results",f"results_logic_{repr}.csv"))
    os.remove(os.path.join(base_path,"results",f"results_logic_{repr}.csv"))
    total_results = pd.concat([total_results,results])
total_results.to_csv(os.path.join(base_path,"results","results_logic.csv"),index=False)


# # Run the GNN's on the different representations

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

