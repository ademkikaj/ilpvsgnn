import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
# import kernel_baselines as kb
from auxiliarymethods.kernel_evaluation import kernel_svm_evaluation, kernel_svm_evaluation_repetitions,linear_svm_evaluation
import torch
import os
from grakel import GraphKernel
from grakel.graph import Graph
from grakel.kernels import ShortestPath, SvmTheta, GraphletSampling,weisfeiler_lehman, RandomWalk
from torch_geometric.utils import to_dense_adj
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import string
import itertools


def main():
    datasets = ["krk","bongard","train","mutag","nci","cancer"]
    datasets = ["krk","bongard","train","sameGen","cyclic"]
    datasets = ["bongard","sameGen","cyclic"]
    datasets = ["financial"]

    #datasets = ["mutag","nci","cancer","financial","PTC"]
    #datasets = ["mutag","nci","cancer","train","krk"]
    #datasets = ["cyclic"]
    representations = ["node_only","node_edge","edge_based","Klog"]
    #representations = ["node_only","node_edge","edge_based"]
    representations = ["node_edge","Klog"]
    C =[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]
    C = [10**5,10**4,10 ** 3, 10 ** 2,10 ** 1]
    C = [10 ** 1, 10 ** 0, 10 ** -1]
    #C = [10 **1, 10**0] 
    #C = [10**1]
    svm_kernel = ["linear","precomputed","rbf","poly"]
    svm_kernel = ["precomputed"]
    kernel = ["shortest_path","weisfeiler_lehman","graphlet_sampling","random_walk"]
    kernel = ["shortest_path","weisfeiler_lehman","graphlet_sampling"]
    #kernel = ["random_walk"]
    for dataset in datasets:
        # result columns: dataset, representation, graph_kernel,C,train_accuracy, test_accuracy
        results = pd.DataFrame()
        for repr in representations:
            #print(f"Running {dataset} with {repr}")
            # create the encoder for this representation
            path = os.path.join("docker","Benchmark",dataset,repr,"graph","train.pt")
            train_graphs = torch.load(path)
            path = os.path.join("docker","Benchmark",dataset,repr,"graph","test.pt")
            test_graphs = torch.load(path)
            print("Creating node encoder")
            node_encoder = create_node_encoder(train_graphs,test_graphs)
            print("Node encoder created")
            for c in C:
                for svm_k in svm_kernel:
                    for k in kernel:
                        print(f"Running {dataset} with {repr} and {k} kernel")
                        new_results = run_kernel(dataset,repr,k,svm_k,c,node_encoder)
                        #print(new_results)
                        results = pd.concat([results,new_results])

        # write the results to a csv file
        output_path = f"docker/Benchmark/{dataset}/results/kernel_results.csv"
        results.to_csv(output_path,index=False)
    print(results)
    return


def run_kernel(dataset,repr,kernel,svm_kernel,C,node_encoder):

    path = os.path.join("docker","Benchmark",dataset,repr,"graph","train.pt")
    train_graphs = torch.load(path)
    path = os.path.join("docker","Benchmark",dataset,repr,"graph","test.pt")
    test_graphs = torch.load(path)

    
    #node_encoder = create_node_encoder(train_graphs,test_graphs)
    G_train = []
    y_train = [g.y.item() for g in train_graphs]
    for g in train_graphs:
        adjacency = to_dense_adj(g.edge_index).squeeze(0)
        adjacency = adjacency.int().numpy()
        # optional addition -> add node labels
        # for the node labels, take the index of the max value in the tensor
        node_labels = {i: node_encoder[tuple(g.x[i].tolist())] for i in range(g.x.size(0))}
        graph = Graph(initialization_object=adjacency, node_labels=node_labels)
        G_train.append(graph)
    
    G_test = []
    y_test = [g.y.item() for g in test_graphs]
    for g in test_graphs:
        adjacency = to_dense_adj(g.edge_index).squeeze(0)
        adjacency = adjacency.int().numpy()
        # optional addition -> add node labels
        # for the node labels, take the index of the max value in the tensor
        node_labels = {i: node_encoder[tuple(g.x[i].tolist())] for i in range(g.x.size(0))}
        graph = Graph(initialization_object=adjacency, node_labels=node_labels)
        G_test.append(graph)
    
    # intialize the graph kernel
    if kernel == "shortest_path":
        graph_kernel = ShortestPath(normalize=False,with_labels=False)
    elif kernel == "graphlet_sampling":
        graph_kernel = GraphletSampling(normalize=False)
    elif kernel == "weisfeiler_lehman":
        graph_kernel = weisfeiler_lehman.WeisfeilerLehman(n_jobs=3,normalize=False)
    elif kernel ==  "random_walk":
        graph_kernel = RandomWalk(n_jobs=3,normalize=False)
    
    # fit the kernel
    K_train = graph_kernel.fit_transform(G_train)
    K_test = graph_kernel.transform(G_test)
    
    # train the SVM
    clf = SVC(C=C,kernel=svm_kernel)
    clf.fit(K_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(K_train))

    # test the SVM
    y_pred = clf.predict(K_test)
    test_acc = accuracy_score(y_test, y_pred)

    result = pd.DataFrame({
        "dataset": [dataset],
        "representation": [repr],
        "graph_kernel": [kernel],
        "C": [C],
        "svm_kernel": [svm_kernel],
        "train_accuracy": [train_acc],
        "test_accuracy": [test_acc]
    },index=[0])

    return result

def create_node_encoder(graphs,graphs_test):
    encoder = {}
    letter_iter = iter(itertools.count(1))
    graphs = graphs + graphs_test
    for graph in graphs:
        for row in graph.x:
            row_tuple = tuple(row.tolist())
            if row_tuple not in encoder:
                encoder[row_tuple] = next(letter_iter)
    return encoder





if __name__ == "__main__":
    main()
