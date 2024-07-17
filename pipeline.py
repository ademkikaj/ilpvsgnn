import subprocess
import shlex
from bongardGenerator.bongardGenerator import generate_bongard_example
import re
import pandas as pd
import docker
from datasets.Bongard.toRelational import toRelational
from Benchmark.bongard.toGraph import toGraph
import random
from torch_geometric.loader import DataLoader
from gnn_baselines.gnn_architectures import *
import time
from tqdm import tqdm
from datasets.Bongard.BongardDataset import BongardDataset

# Variables
num_examples = 100
object_complexity = 20
relation_complexity = 10

# to do variables
test_size = 0.2

def extract_results(results_file:str) -> pd.DataFrame:
    # Read the file
    with open(results_file, 'r') as file:
        lines = file.readlines()
    
    overall_metrics = {}
    start = lines.index(" **/\n")
    
    overall_metrics['time_discretization'] = float(lines[start+2].split()[-1])
    overall_metrics['time_induction'] = float(lines[start+3].split()[-1])

    # Extract the training metrics
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

    # Extract the testing metrics
    testing_start = training_end + 1
    testing_end = lines.index('Compact notation of tree:\n') - 2
    testing_metrics = {}
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

def run_experiment():
    # Start the docker container and run the bash script
    client = docker.from_env()
    container = client.containers.get('ace_system')
    if container.status != 'running':
        container.start()
    dir = "/user"
    start = time.time()
    result = container.exec_run('bash run_ace.sh',workdir=dir)
    end = time.time()
    return end-start 

def generate_relational():
    # convert logic to relational
    bongard_relations = {
        'bongard': ['problemId','class'],
        'square': ['problemId','objectId'],
        'circle': ['problemId','objectId'],
        'triangle': ['problemId','objectId'],
        'in': ['problemId','objectId1','objectId2']
    }
    relational_converter = toRelational('docker/Bongard/Experiment/bongard.kb','docker/Bongard/Experiment/Relational/',bongard_relations)
    relational_converter.logic_to_relational()

def generate_graph():
    # convert relational to graph
    graph_converter = toGraph('docker/Bongard/Experiment/Relational/', 'docker/Bongard/Experiment/Graph/')
    graphs = graph_converter.NodeOnlyRepresentation()
    torch.save(graphs, 'datasets/Bongard/Experiment/raw/datalist.pt')
    return graphs

def train(train_loader, model, optimizer, device):
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
def test(loader, model, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def gnn_training(graphs) -> pd.DataFrame:
    # variables
    test_size = 0.2
    val_size = 0.1
    model_name = 'GIN'
    layers = 5
    hidden_dims = 256
    epochs = 40
    lr = 0.1
    min_lr = 0.00001

    num_node_features = 3
    num_classes = 2

    val_index = int(len(graphs) * (1 - val_size - test_size))
    test_index = int(len(graphs) * (1 - test_size))

    train_loader = DataLoader(graphs[:val_index], batch_size=16, shuffle=True)
    val_loader = DataLoader(graphs[val_index:test_index], batch_size=16, shuffle=False)
    test_loader = DataLoader(graphs[test_index:], batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = modelsMetadata().models
    model = models[model_name](num_node_features,num_classes,layers, hidden_dims).to(device)
    model.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5,min_lr=min_lr)

    train_data = pd.DataFrame()

    # Time the training of the GNN
    start = time.time()
    for epoch in tqdm(range(epochs),desc="Epochs"):
        lr = scheduler.optimizer.param_groups[0]['lr']
        train_loss = train(train_loader, model, optimizer, device)

        val_acc = test(val_loader, model, device)
        scheduler.step(val_acc)

        train_acc = test(train_loader, model, device)
        test_acc = test(test_loader, model, device)

        train_data = pd.concat([
            train_data,
            pd.DataFrame({'train_acc':train_acc,'train_loss':train_loss,'val_acc':val_acc,'test_acc':test_acc,'lr':lr,'epoch':epoch},index=[0])
            ])
    end = time.time()
    
    train_data['model'] = model
    train_data['layers'] = layers
    train_data['hidden_dims'] = hidden_dims
    train_data['epochs'] = epochs
    train_data['runtime'] = end-start
    train_data['total_examples'] = len(graphs)
    train_data['test_size'] = int(test_size * len(graphs))
    train_data['val_size'] = int(val_size * len(graphs))
    train_data['train_size'] = len(graphs) - train_data['test_size'] - train_data['val_size']
    # maybe add number of positive examples and negative examples
    
    return train_data

def scaling_experiment():
    # Run scaling experiment
    logic_results = pd.DataFrame()
    gnn_results = pd.DataFrame()
    for i in range(10,200,10):
        # Generate the bongard logic problem
        num_examples = i
        object_complexity = 10
        relation_complexity = 5
        filePath = "docker/Bongard/Experiment/bongard.kb"
        generate_bongard_example(num_examples,object_complexity,relation_complexity,None,filePath)
        
        # Run the logic side of the experiment
        run_experiment()
        new_df = extract_results("docker/Bongard/Experiment/tilde/bongard.out")
        new_df['object_complexity'] = object_complexity
        new_df['relation_complexity'] = relation_complexity
        logic_results = pd.concat([logic_results,new_df])

        # convert the logic program to a relational data
        generate_relational()
        # convert the relational data to graph data
        graphs = generate_graph()

        new_gnn = gnn_training(graphs)
        new_gnn['object_complexity'] = object_complexity
        new_gnn['relation_complexity'] = relation_complexity
        gnn_results = pd.concat([gnn_results,new_gnn])

    # save the results
    logic_results.to_csv("results/bongard_logic_scaling.csv", index=False)
    gnn_results.to_csv("results/bongard_gnn_scaling.csv", index=False)
    
def complexity_experiment():
    # Run complexity experiments
    results = pd.DataFrame()
    for i in range(5,45,5):
        num_examples = 100
        object_complexity = i
        relation_complexity = 5
        filePath = "docker/Bongard/Experiment/bongard.kb"
        generate_bongard_example(num_examples,object_complexity,relation_complexity,filePath)
        run_experiment()
        new_df = extract_results("docker/Bongard/Experiment/tilde/bongard.out")
        new_df['object_complexity'] = object_complexity
        new_df['relation_complexity'] = relation_complexity
        results = pd.concat([results,new_df])

    # save the results
    results.to_csv("results/bongard_complexity.csv", index=False)



if __name__ == "__main__":
    scaling_experiment()
    #complexity_experiment()

    # generate_bongard_example(200,3,2,"docker/Bongard/Experiment/bongard.kb")
    # # generate relational data
    # generate_relational()
    # # generate graph data
    # graphs = generate_graph()




    
