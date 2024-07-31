import os
import pandas as pd
import re
import torch
import docker 
import time
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.nn import functional as F
import shutil
import random
import yaml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np
import sys
from pyswip import Prolog
from sklearn.model_selection import train_test_split
import subprocess

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)


from Benchmark.tilde import Tilde
from Benchmark.popper_system import Popper
from Benchmark.aleph_system import Aleph

# needs to be generalised
from Benchmark.bongard.toGraph import toGraph
from Benchmark.bongard.toLogic import toLogic
from gnn_baselines.gnn_architectures import modelsMetadata
from Benchmark.toILP import toILP

from popper.util import Settings, print_prog_score, format_prog
from popper.loop import learn_solution
import multiprocessing as mp

TILDE = True
ALEPH = False
POPPER = False


class Experiment:

    def __init__(self,path,logic=False,relational=False,graph=False,dataset_name=None,gnn_config_path=None,relations=None):
        # Experiment is a class that hold the paths to directories where all related files will be stored for performing the experiment
        self.original_path = path
        self.base_path = os.path.join("docker",path)
        self.logic_path = os.path.join(self.base_path,"logic")
        self.graph_path = os.path.join(self.base_path,"graph")
        self.graph_test_path = os.path.join(self.graph_path,"test")
        self.relational_path = os.path.join(self.base_path,"relational")
        self.relational_test_path = os.path.join(self.relational_path,"test")
        self.relational_train_path = os.path.join(self.relational_path,"train")
        self.results_path = os.path.join(self.base_path,"results")
        self.representations = ["node_only","node_edge","edge_based","Klog"]
        self.make_directories()
        self.toILP = toILP(self.relational_path,self.logic_path,dataset_name)
        self.toGraph = toGraph

        if gnn_config_path is not None:
            with open(gnn_config_path) as file:
                self.config = yaml.safe_load(file)
            
        # index is used to keep track of the experiment iteration number, determine index by looking at results directory
        # get all the files ending with i.csv
        
        
        logic_results, gnn_results = self.get_all_step_results()
        self.index = min(len(logic_results),len(gnn_results))
        

        # Flags to indicate which data is available
        self.logic = logic
        self.graph = graph
        self.relational = relational

        self.dataset_name = dataset_name

        # copy the bongard settings and bg file to the logic directory
        shutil.copy("docker/Bongard/Experiment/bongard.s",self.logic_path+"/tilde")
        shutil.copy("docker/Bongard/Experiment/bongard.bg",self.logic_path+"/tilde")

        self.relations = relations

    

    def make_directories(self):
        paths = [self.base_path,self.logic_path,self.graph_path,self.relational_path,self.results_path,self.graph_test_path,self.relational_test_path,self.relational_train_path]
        # add training relational path and testing relational path
        for repr in self.representations:
            paths.append(os.path.join(self.base_path,repr))
            paths.append(os.path.join(self.base_path,repr,"graph"))
            paths.append(os.path.join(self.base_path,repr,"logic"))
            paths.append(os.path.join(self.base_path,repr,"logic","aleph"))
            paths.append(os.path.join(self.base_path,repr,"logic","popper"))
            paths.append(os.path.join(self.base_path,repr,"logic","tilde")) 
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
        
        return
    
    def get_all_step_results(self):
        logic_results = []
        gnn_results = []
        for file in os.listdir(self.results_path):
            if not file.endswith("combined.csv"):
                if "logic" in file:
                    logic_results.append(file)
                else:
                    gnn_results.append(file)
        return logic_results, gnn_results
    
    def logic_to_relational(self):
        result_path = os.path.join(self.logic_path,f"{self.dataset_name}.kb")
        with open(result_path,'r') as file:
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
            output_path = os.path.join(self.relational_path,pred+'.csv')
            df[pred].to_csv(output_path,index=False)
        
        return

    def run_ilp(self):
        # Start the docker container and run the bash script
        client = docker.from_env()
        container = client.containers.get('ace_system')
        if container.status != 'running':
            container.start()
        dir = "/user"
        start = time.time()
        # bash script requires the path to the logic data
        print(self.original_path)
        result = container.exec_run(f"bash run_ace.sh {self.original_path}/logic/tilde",workdir=dir)
        end = time.time()
        self.extract_tilde_program()
        return end-start 

    def run_popper(self):
        popper_path = os.path.join(self.logic_path,"popper")
        settings = Settings(kbpath=popper_path,timeout=600,datalog=True,quiet=True)
        prog, score, stats = learn_solution(settings)

        if settings.show_stats:
            stats.show()

        popper_results = pd.DataFrame()

        if score is not None:
            tp, fn, tn, fp, size = score
            
            precision = 'n/a'
            if (tp+fp) > 0:
                precision = f'{tp / (tp+fp):0.2f}'
            
            recall = 'n/a'
            if (tp+fn) > 0:
                recall = f'{tp / (tp+fn):0.2f}'
            
            train_acc = (tp+tn) / (tp+tn+fp+fn)
        else:
            train_acc = None
        return prog,train_acc

    def popper_hyporthesis(self,prog):
        new_program = []
        
        new_program.append(":- style_check(-singleton).")

        new_clause = prog.replace(f"{self.dataset_name}(A)",f"{self.dataset_name}(A,pos)")
        new_clause = new_clause.replace(".", ", !.")

        new_program.append(new_clause)

        # add the negative clause
        new_program.append(f"{self.dataset_name}(A,neg).")

        # write to file
        with open(self.logic_path + f"/popper/{self.dataset_name}_out_{self.index}.pl", "w") as file:
            for line in new_program:
                file.write(line + "\n")
        
        return 
    

    def extract_tilde_program(self):

        result_path = os.path.join(self.logic_path, f"tilde/tilde/{self.dataset_name}.out")
        with open(result_path, 'r') as file:
            lines = file.readlines()
        
        start = lines.index("Equivalent prolog program:\n")
        program = []
        index = start + 2
        while lines[index].strip():
            program.append(lines[index])
            index += 2
        
        # map the [pos] and [neg] to pos and neg
        for i in range(len(program)):
            program[i] = program[i].replace("[pos]","pos")
            program[i] = program[i].replace("[neg]","neg")

        # add the styll check singleton
        program.insert(0,":- style_check(-singleton).\n")
        
        # write to a file
        with open(self.logic_path + f"/tilde/{self.dataset_name}_out_{self.index}.pl", "w") as file:
            for line in program:
                file.write(line)
        return
    
    # Extract tilde results
    def extract_results(self,train=True,test=False) -> pd.DataFrame:
        results_file = os.path.join(self.logic_path,f"tilde/tilde/{self.dataset_name}.out")
        with open(results_file, 'r') as file:
            lines = file.readlines()
        
        overall_metrics = {}
        start = lines.index(" **/\n")
        
        overall_metrics['time_discretization'] = float(lines[start+2].split()[-1])
        overall_metrics['time_induction'] = float(lines[start+3].split()[-1])

        # Extract the training metrics
        if train:
            training_start = lines.index("Training:\n") + 1
            training_end = lines.index("Testing:\n")
            training_metrics = {}
            training_section = lines[training_start:training_end]

            training_metrics['n_examples'] = int(training_section[0].split()[-1])
            training_metrics['n_pos_examples'] = int(training_section[4].split()[-1])
            training_metrics['n_neg_examples'] = int(training_section[5].split()[-1])
            
            training_metrics['accuracy'] = float(training_section[9].split()[1])
            training_metrics['accuracy_std'] = float(training_section[9].split()[3].strip(','))
            training_metrics['accuracy_default'] = float(training_section[9].split()[-1].strip(')'))
            training_metrics['cramers_coeff'] = float(training_section[10].split()[2])
            training_metrics['TP'] = float(training_section[11].split()[4].strip(','))
            training_metrics['FN'] = float(training_section[11].split()[-1])
            training_metrics['type'] = 'training'

        testing_metrics = {}
        if test:
            # Extract the testing metrics
            testing_start = training_end + 1
            testing_end = lines.index('Compact notation of tree:\n') - 2
            testing_section = lines[testing_start:testing_end]

            testing_metrics['n_examples'] = int(testing_section[0].split()[-1])
            testing_metrics['n_pos_examples'] = int(testing_section[4].split()[-1])
            testing_metrics['n_neg_examples'] = int(testing_section[5].split()[-1])

            testing_metrics['accuracy'] = float(testing_section[9].split()[1])
            testing_metrics['accuracy_std'] = float(testing_section[9].split()[3].strip(','))
            testing_metrics['accuracy_default'] = float(testing_section[9].split()[-1].strip(')'))
            testing_metrics['cramers_coeff'] = float(testing_section[10].split()[2])
            testing_metrics['TP'] = float(testing_section[11].split()[4].strip(','))
            testing_metrics['FN'] = float(testing_section[11].split()[-1])
            testing_metrics['type'] = 'testing'

            training_metrics['total_examples'] = training_metrics['n_examples'] + testing_metrics['n_examples']
            testing_metrics['total_examples'] = training_metrics['n_examples'] + testing_metrics['n_examples']

        # time metrics
        training_metrics['time_total'] = float(overall_metrics['time_discretization']) + float(overall_metrics['time_induction'])
        testing_metrics['time_total'] = float(overall_metrics['time_discretization']) + float(overall_metrics['time_induction'])
        
        df = pd.DataFrame([training_metrics, testing_metrics])
        # should be written to the results directory: Tilde and some identifier
        return df

    def aleph_hypothesis(self, model):
        program = model.hypothesis
        # add an extra argument to the program clauses
        new_program = []

        # add style check singleton
        new_program.append(":- style_check(-singleton).")
        for clause in program:
            clause = clause.replace(".", ", !.")
            new_program.append(clause.replace(f"{self.dataset_name}(A)",f"{self.dataset_name}(A,pos)"))
        # add the negative clause
        new_program.append(f"{self.dataset_name}(A,neg).")
        
        with open(self.logic_path + f"/aleph/{self.dataset_name}_out_{self.index}.pl", "w") as file:
            for line in new_program:
                file.write(line + "\n")
        return
        
    
    def test_program(self,program,result):
        # test the program on the test set
        prolog = Prolog()
 

        # load the program
        prolog.consult(program)

        # convert to prolog file
        # shutil.copy(self.logic_path + f"/tilde/{self.dataset_name}.kb",self.logic_path + f"/tilde/{self.dataset_name}.pl")


        # load the test background knowledge
        self.toILP.toProlog()
        prolog.consult(self.logic_path + f"/{self.dataset_name}.pl")
        
        # load the test queries
        df = pd.read_csv(self.relational_path + f"/test/{self.dataset_name}.csv")

        
        total = 0
        correct = 0
        for _, row in df.iterrows():
            # create the query
            query = f"{self.dataset_name}({row['problemId']},Result)"
            true_value = row['class']
            query_result = list(prolog.query(query))
            if query_result:
                if query_result[0]['Result'] == true_value:
                    correct += 1
                total += 1
        
        result.put(correct/total)
        return 

        query_result = list(prolog.query(f"{self.dataset_name}(12,Result)"))

        if query_result:
            return query_result[0]['Result']
        else:
            return None




    
    # One training epoch for GNN model.
    def train(self,train_loader, model, optimizer, device):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        return total_loss  

    # Get acc. of GNN model.
    def test(self,loader, model, device):
        model.eval()

        correct = 0
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

    def stratify_data(self,graphs,test_size,val_size):
        n_test = int(test_size * len(graphs))
        n_val = int(val_size * len(graphs))
        n_train = len(graphs) - n_test - n_val

        n_train_pos = int(n_train / 2)
        n_train_neg = n_train - n_train_pos

        n_val_pos = int(n_val / 2)
        n_val_neg = n_val - n_val_pos

        n_test_pos = int(n_test / 2)
        n_test_neg = n_test - n_test_pos

        train_list = []
        val_list = []
        test_list = []

        for graph in graphs:
            if graph.y == torch.tensor([1]):
                if n_train_pos > 0:
                    train_list.append(graph)
                    n_train_pos -= 1
                elif n_val_pos > 0:
                    val_list.append(graph)
                    n_val_pos -= 1
                else:
                    test_list.append(graph)
                    n_test_pos -= 1
            else:
                if n_train_neg > 0:
                    train_list.append(graph)
                    n_train_neg -= 1
                elif n_val_neg > 0:
                    val_list.append(graph)
                    n_val_neg -= 1
                else:
                    test_list.append(graph)
                    n_test_neg -= 1
        return train_list, val_list, test_list
    
    def gnn_training(self,graphs,test_graphs,model,layers,hidden_dims) -> pd.DataFrame:
        # no use of the folds, just multiple repetitions

        # variables
        model_name = model
        layers = layers
        hidden_dims = hidden_dims
        
        epochs = 200
        
        min_lr = 0.00001
        batch_size = 16

        # folds = self.config['folds']
        repititions = self.config['repetitions']

        num_node_features = graphs[0].num_node_features
        num_classes = 2
        num_edge_features = graphs[0].num_edge_features
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        test_accuracies = []
        runtimes = []

        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=True)

        for i in range(repititions):
            print("Repetition: ", i+1)

            random.shuffle(graphs)
            train_index, val_index = train_test_split(range(len(graphs)),test_size=0.1)
                
            best_val_acc = 0.0
            best_test_acc = 0.0

            train_dataset = [graphs[i] for i in train_index]
            val_dataset = [graphs[i] for i in val_index]

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            models = modelsMetadata().models
            model = models[model_name](num_node_features,num_edge_features,num_classes,layers, hidden_dims).to(device)
            model.reset_parameters()

            lr = 0.01
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5,min_lr=min_lr)

            train_data = pd.DataFrame()
            # Time the training of the GNN
            start = time.time()
            for epoch in tqdm(range(epochs),desc="Epochs"):
                lr = scheduler.optimizer.param_groups[0]['lr']
                train_loss = self.train(train_loader, model, optimizer, device)
                val_acc = self.test(val_loader, model, device)
                scheduler.step(val_acc)
                train_acc = self.test(train_loader, model, device)
                test_acc = self.test(test_loader, model, device)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                
                if lr <= min_lr:
                    break

            end = time.time()

            runtimes.append(end-start)

            test_accuracies.append(best_test_acc)
        
        # test accuracies over all the repetitions
        test_acc = np.array(test_accuracies).mean()
        test_acc_std = np.array(test_accuracies).std()

        runtime = np.array(runtimes).mean()

        train_data = pd.DataFrame({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'test_acc_std': test_acc_std,
            'lr': lr,
            'epoch': epoch,
            'model': model,
            'layers': layers,
            'hidden_dims': hidden_dims,
            'total_epochs': epochs,
            'runtime': runtime,
            'test_size': len(test_loader),
            'val_size': len(val_dataset),
            'train_size': len(train_dataset)
            },index=[0])
        
        return train_data



    def run(self):
        # Following steps are performed in the experiment:

        self.run_logic()

        # 2. Convert relational data to graph or logic format (Use of certain conversion method)
        representations = {}
        test_representations = {}
        if not self.graph:
            representation_names = self.config['representations']
            for name in representation_names:
                string = f"toGraph(self.relational_path).{name}()"
                test_string = f"toGraph(self.relational_path + '/test').{name}()"
                representations[name] = eval(string)
                test_representations[name] = eval(test_string)
                if self.index == 0:
                    torch.save(test_representations[name],os.path.join(self.graph_test_path,f"{name}.pt"))
                torch.save(representations[name],os.path.join(self.graph_path,f"{name}_{self.index}.pt"))

        
        # 5. Run the GNN models (GIN, GCN, etc)
        total_runs = len(self.config['models']) * len(self.config['layers']) * len(self.config['hidden_dims']) * len(representations.keys()) * self.config['repetitions']
        done_runs = 0
        total_gnn_data = pd.DataFrame()
        for model in self.config['models']:
            for layers in self.config['layers']:
                for hidden_dims in self.config['hidden_dims']:
                    for representation in representations.keys():
                        print(f"Run {done_runs}/{total_runs}")
                        graphs = representations[representation]
                        test_graphs = test_representations[representation]
                        result = self.gnn_training(graphs,test_graphs,model,layers,hidden_dims)
                        result['representation'] = representation
                        total_gnn_data = pd.concat([total_gnn_data,result])
                        done_runs += self.config['repetitions']

                    

        # 6. Save the results
        total_gnn_data.to_csv(os.path.join(self.results_path,f"gnn_results_{self.index}.csv"),index=False)

        # rename all the relational data and move to relational train directory
        for file in os.listdir(self.relational_path):
            if file.endswith(".csv"):
                name = file[:-4]
                shutil.move(self.relational_path + f"/{file}",self.relational_train_path +f"/{name}_{self.index}.csv")
        
        self.index += 1
        return
    
    def run_logic(self):

        graph_converter = toGraph(relational_path=self.relational_path)
        graph_converter_test = toGraph(relational_path=self.relational_test_path)

        logic_converter = toLogic()

        for repr in self.representations:
            # build the graph representations
            string = f"graph_converter.{repr}()"
            test_string = f"graph_converter_test.{repr}()"
            graphs = eval(string)
            graphs_test = eval(test_string)

            # write the graphs to the graph directory
            torch.save(graphs, os.path.join(self.base_path, repr, "graph", "train.pt"))
            torch.save(graphs_test, os.path.join(self.base_path, repr, "graph", "test.pt"))

            # convert the graphs to logic
            output_path = os.path.join(self.base_path, repr, "logic",self.dataset_name + ".kb")
            string = f"logic_converter.{repr}(graphs,'{output_path}')"
            eval(string)
            output_path_test = os.path.join(self.base_path, repr, "logic",self.dataset_name + "_test.kb")
            string = f"logic_converter.{repr}(graphs_test,'{output_path_test}')"
            eval(string)

            # remove the truth labels from the test file
            with open(output_path_test, "r") as file:
                lines = file.readlines()
            new_test = []
            for line in lines:
                if self.dataset_name not in line:
                    new_test.append(line)
            with open(output_path_test, "w") as file:
                for line in new_test:
                    file.write(line)
        
        for repr in self.representations:

            ilpConverter = toILP(relational_path=self.relational_path, logic_path=os.path.join(self.base_path, repr, "logic"), dataset_name=self.dataset_name)
            
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
                    "warmode(triangle(+id,+-object)).\n",
                    "warmode(square(+id,+-object)).\n",
                    "warmode(circle(+id,+-object)).\n",
                    "warmode(in(+id,+-object)).\n",
                    "warmode(edge(+id,+-object,+-object)).\n",
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
                    "warmode(triangle(+P,+-S)).\n",
                    "warmode(square(+P,+-S)).\n",
                    "warmode(circle(+P,+-S)).\n",
                    "warmode(edge(+P,+S1,+-S2)).\n",
                    "typed_language(yes).\n",
                    "type(bongard(pic,class)).\n",
                    "type(triangle(pic,obj)).\n",
                    "type(square(pic,obj)).\n",
                    "type(circle(pic,obj)).\n",
                    "type(edge(pic,obj,obj)).\n",
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
            
            ilpConverter.logicToPopper(logic_file_path=os.path.join(self.base_path, repr, "logic", self.dataset_name + ".kb"), label = self.dataset_name ,bias_given=popper_bias)
            ilpConverter.logicToTilde(logic_file_path=os.path.join(self.base_path, repr, "logic", self.dataset_name + ".kb"),givesettings=tilde_settings)
            ilpConverter.logicToAleph(logic_file_path=os.path.join(self.base_path, repr, "logic", self.dataset_name + ".kb"),label= self.dataset_name,given_settings=aleph_settings)


        for repr in self.representations:
            print("Representation: ",repr)
            results = pd.DataFrame()
            target = "class"
            if TILDE:
                tilde = Tilde(dataset_name=self.dataset_name, relational_path=self.relational_path, target=target)
                tilde_results = tilde.run(tilde_input_path=os.path.join(self.base_path,repr))
                tilde_results['representation'] = repr
                results = pd.concat([results,tilde_results])
            if POPPER:
                popper = Popper(name=self.dataset_name,relational_path=self.relational_path,target=target)
                popper_results = popper.run(representation_path=os.path.join(self.base_path,repr))
                popper_results['representation'] = repr
                results = pd.concat([results,popper_results])
            if ALEPH:
                aleph = Aleph(name=self.dataset_name, relational_path=self.relational_path,target=target)
                aleph_results = aleph.run(representation_path=os.path.join(self.base_path,repr))
                aleph_results['representation'] = repr
                results = pd.concat([results,aleph_results])

            results.to_csv(os.path.join(self.base_path,"results",f"results_logic_{repr}.csv"),index=False)

        total_results = pd.DataFrame()
        for repr in self.representations:
            results = pd.read_csv(os.path.join(self.base_path,"results",f"results_logic_{repr}.csv"))
            os.remove(os.path.join(self.base_path,"results",f"results_logic_{repr}.csv"))
            total_results = pd.concat([total_results,results])
        total_results.to_csv(os.path.join(self.base_path,"results",f"logic_results_{self.index}.csv"),index=False)
        
       
        return
    
    def run_gnn(self):
        
        representations = {}
        if not self.graph:
            graphs = toGraph(self.relational_path).NodeOnlyRepresentation()
            representations['node_only'] = graphs
            torch.save(graphs,os.path.join(self.graph_path,f"node_only_{self.index}.pt"))

            graphs = toGraph(self.relational_path).NoFrameRepresentation()
            representations['no_frame'] = graphs
            torch.save(graphs,os.path.join(self.graph_path,f"no_frame_{self.index}.pt"))
            
            graphs = toGraph(self.relational_path).EdgeNodeRepresentation()
            representations['edge_node'] = graphs
            torch.save(graphs,os.path.join(self.graph_path,f"edge_node_{self.index}.pt"))
            
            graphs = toGraph(self.relational_path).KlogRepresentation()
            representations['klog'] = graphs
            torch.save(graphs,os.path.join(self.graph_path,f"klog_{self.index}.pt"))
        
        # 5. Run the GNN models (GIN, GCN, etc)
        total_gnn_data = pd.DataFrame()
        for model in self.config['models']:
            for layers in self.config['layers']:
                for hidden_dims in self.config['hidden_dims']:
                    for representation in representations.keys():
                        graphs = representations[representation]
                        result = self.gnn_training(graphs,model,layers,hidden_dims)
                        result['representation'] = representation
                        total_gnn_data = pd.concat([total_gnn_data,result])
        
        # 6. Save the results
        total_gnn_data.to_csv(os.path.join(self.results_path,f"gnn_results_{self.index}.csv"),index=False)
        self.index += 1
        return
    
    def run_representations(self,repr):
        # the representation are already loaded in their respective directories

        # first thing is to conver to the appropriate logic formats
        # popper
        

        graphs = f"toGraph(self.relational_path).{repr}()"
        test_graphs = f"toGraph(self.relational_path + '/test').{repr}()"

        total_runs = len(self.config['models']) * len(self.config['layers']) * len(self.config['hidden_dims']) * self.config['repetitions']
        done_runs = 0
        total_gnn_data = pd.DataFrame()
        for model in self.config['models']:
            for layers in self.config['layers']:
                for hidden_dims in self.config['hidden_dims']:
                    print(f"Run {done_runs}/{total_runs}")
                    result = self.gnn_training(graphs,test_graphs,model,layers,hidden_dims)
                    result['representation'] = repr
                    total_gnn_data = pd.concat([total_gnn_data,result])
                    done_runs += self.config['repetitions']
        
        # 6. Save the results
        total_gnn_data.to_csv(os.path.join(self.results_path,f"gnn_results_{repr}.csv"),index=False)

        return

    
    
    def combine_results(self,logic=True,gnn=True):
        # combine all the gnn_results_i and tilde_results_i into 2 dataframes
        if logic:
            logic_results = pd.DataFrame()
            for i in range(self.index):
                logic_results = pd.concat([logic_results,pd.read_csv(os.path.join(self.results_path,f"logic_results_{i}.csv"))])
            logic_results.to_csv(os.path.join(self.results_path,"logic_results_combined.csv"),index=False)
        if gnn:
            gnn_results = pd.DataFrame()
            for i in range(self.index):
                gnn_results = pd.concat([gnn_results,pd.read_csv(os.path.join(self.results_path,f"gnn_results_{i}.csv"))])
            gnn_results.to_csv(os.path.join(self.results_path,"gnn_results_combined.csv"),index=False)
        return
    
    def log_extra_metrics(self,col,value,index,logic=True,gnn=True):
        # load the current results
        if logic:
            logic_results = pd.read_csv(os.path.join(self.results_path,f"logic_results_{index}.csv"))
            logic_results[col] = value
            logic_results.to_csv(os.path.join(self.results_path,f"logic_results_{index}.csv"),index=False)
        if gnn:
            gnn_results = pd.read_csv(os.path.join(self.results_path,f"gnn_results_{index}.csv"))
            gnn_results[col] = value
            gnn_results.to_csv(os.path.join(self.results_path,f"gnn_results_{index}.csv"),index=False)
        return

    def split_relational_data(self):
        # load the main data
        df = pd.read_csv(self.relational_path + f"/{self.dataset_name}.csv")
        train, test = train_test_split(df,test_size=0.2)
        train_problem_ids = train['problemId']
        
        kb = {}
        # load all csv file in relational directory
        for file in os.listdir(self.relational_path):
            if file.endswith(".csv"):
                kb[file[:-4]] = pd.read_csv(self.relational_path + f"/{file}")

        train = {}
        test = {}
        for key in kb.keys():
            df = kb[key]
            train[key] = df[df['problemId'].isin(train_problem_ids)]
            test[key] = df[~df['problemId'].isin(train_problem_ids)]
        
        # write them away
        for key in train.keys():
            train[key].to_csv(self.relational_path + f"/{key}.csv",index=False)
            test[key].to_csv(self.relational_test_path + f"/{key}.csv",index=False)
