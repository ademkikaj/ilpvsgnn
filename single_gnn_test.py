import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation_edit import gnn_evaluation, gnn_evaluation_wandb, train_wandb
from gnn_baselines.gnn_architectures import GIN, GINE, GINEWithJK, GINWithJK, GCN, GraphSAGE, ASAP,TopK,SAGPool, GlobalAttentionNet, HGT, GNNHetero
from datasets.Bongard.BongardDataset import BongardDataset
import json
from utils import dotdict
import numpy as np
import wandb
from torch_geometric.nn.models import GAT
from torch_geometric.nn import to_hetero


def stratify_dataset(dataset, ratio_pos, length):
    indices = []
    amount_pos = int(ratio_pos * length)
    amount_neg = length - amount_pos
    for i in range(len(dataset)):
        pos = True if dataset.get(i).y == 1 else False
        if pos and amount_pos > 0:
            amount_pos -= 1
        elif not pos and amount_neg > 0:
            amount_neg -= 1
        else:
            indices.append(i)
    return indices

def initResults(args):
    results = {}
    for arg in vars(args):
        results[arg] = getattr(args, arg)
    return results


def createResultsJson(acc,s_1,s_2):
    result = {"Accuracy": acc, "std all" : s_1, "std complete" : s_2}
    return result


def sizeExperiment(args):
    size_results_gin = {}
    for i in range(50, 251, 50):
        d = BongardDataset(root='datasets/'+args.dataset_name+'/'+args.dataset_type)
        d.process()
        indices = stratify_dataset(d, 0.5, i)
        d.remove_datapoints(indices)
        print("Size: " + str(len(d)))
        acc, s_1, s_2 = gnn_evaluation(args.model, d, args.layers, args.hidden_dims, max_num_epochs=args.max_num_epochs,
                                       batch_size=args.batch_size, start_lr=args.start_lr, num_repetitions=args.num_reps,
                                       all_std=True)
        print("GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        size_results_gin[i] = createResultsJson(acc, s_1, s_2)
    return size_results_gin

def ratioExperiment(args):
    ratio_results_gin = {}
    for i in np.arange(0,1,0.2):
        d = BongardDataset(root='datasets/'+args.dataset_name+'/'+args.dataset_type)
        d.process()
        indices  = stratify_dataset(d, i, 100)
        d.remove_datapoints(indices)

        acc, s_1, s_2 = gnn_evaluation(args.model, d, args.layers,args.hidden_dims, max_num_epochs=args.max_num_epochs, batch_size=args.batch_size,start_lr=args.start_lr, num_repetitions=args.num_reps, all_std=True)
        print("GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        ratio_results_gin[round(i,2)]= createResultsJson(acc,s_1,s_2)
    return ratio_results_gin





def main():
    args = dotdict()

    ### Set the parameters
    args.dataset_name = "Bongard"
    args.dataset_type = "Heterogeneous"
    d = BongardDataset(root='/Users/nicolasdebie/Master thesis/Benchmarking-GNN-ILP/datasets/'+args.dataset_name+'/'+args.dataset_type)
    d.process()
    print(d[0].edge_attrs)

    args.num_reps = 5
    args.layers = [1,2,3,4,5]
    args.hidden_dims = [32,64,128]
    args.max_num_epochs = 200
    args.batch_size = 64
    args.start_lr = 0.01
    args.datasetSize = len(d)
    args.positives = d.get_amount_of_positives()
    args.negatives = d.get_amount_of_negatives()

    results = initResults(args)

    args.model = HGT

    ### Experiments
    # the influence of the length of the dataset with 50% positive examples
    #results["Size results GIN"] = sizeExperiment(args)
    # the influence of positive examples ration in the dataset
    #results["Ratio results GIN"] = ratioExperiment(args)

    ### GNN evaluation
    acc, s_1, s_2 = gnn_evaluation(args.model, d, args.layers, args.hidden_dims, max_num_epochs=args.max_num_epochs,
                                   batch_size=args.batch_size, start_lr=args.start_lr, num_repetitions=args.num_reps,
                                   all_std=True)
    print("GINE " + str(acc) + " " + str(s_1) + " " + str(s_2))
    results["GINE"] = createResultsJson(acc, s_1, s_2)


    # write results to json file
    with open('results/results' + args.dataset_type + '.json', 'w') as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
