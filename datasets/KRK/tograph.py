import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


white_kings = pd.read_csv('datasets/KRK/Relational/white_king.csv')
white_rooks = pd.read_csv('datasets/KRK/Relational/white_rook.csv')
black_kings = pd.read_csv('datasets/KRK/Relational/black_king.csv')
classes = pd.read_csv('datasets/KRK/Relational/class.csv')


total_graphs = 700
print("Total illigal: ",len([i for i in classes['class'] if i == 'illegal']))
print("Total legal: ",len([i for i in classes['class'] if i == 'legal']))

assert len(classes) == len(white_kings) == len(white_rooks) == len(black_kings)



def create_base():
    data_list = []
    for i in range(total_graphs):
        # white king, white rook, black king
        # node attributes
        x = torch.zeros(64, 3)
        # edge indices
        total_edges  = (8*7)*4
        edge_indices = torch.zeros((2, total_edges), dtype=torch.int64)

        # fill node attributes
        white_king = white_kings.iloc[i]
        x_index = white_king['rank']*8 + white_king['file']
        x[x_index][0] = 1

        white_rook = white_rooks.iloc[i]
        x_index = white_rook['rank']*8 + white_rook['file']
        x[x_index][1] = 1

        black_king = black_kings.iloc[i]
        x_index = black_king['rank']*8 + black_king['file']
        x[x_index][2] = 1

        assert white_rook['id'] == white_king['id'] == black_king['id']
        # fill edge indices
        edge_index = 0
        for k in range(8):
            for j in range(8):
                # add horizontal edge
                if j != 7:
                    edge_indices[0][edge_index] = k*8 + j
                    edge_indices[1][edge_index] = k*8 + j + 1
                    edge_index += 1
                    edge_indices[0][edge_index] = k*8 + j + 1
                    edge_indices[1][edge_index] = k*8 + j
                    edge_index += 1

                # add vertical edge
                if k != 7:
                    edge_indices[0][edge_index] = k*8 + j
                    edge_indices[1][edge_index] = (k+1)*8 + j
                    edge_index += 1
                    edge_indices[0][edge_index] = (k+1)*8 + j
                    edge_indices[1][edge_index] = k*8 + j
                    edge_index += 1
        
        # Target y
        y_id = classes.iloc[i]["id"]

        y = torch.tensor([0]) if classes.iloc[i]['class'] == 'illegal' else torch.tensor([1])

        # create default edge features
        edge_attr = torch.ones((edge_indices.shape[1],1),dtype=torch.float)
        # create data object
        data = Data(x=x, edge_index=edge_indices, y=y, edge_attr=edge_attr)
        
        # append to list
        data_list.append(data)
    return data_list

def is_white_king(node):
        possibilities = torch.tensor([[1,0,0],[1,1,0],[1,0,1],[1,1,1]],dtype=torch.int64)
        for row in possibilities:
            if torch.equal(node,row):
                return True
        return False

def is_black_king(node):
        possibilities = torch.tensor([[0,0,1],[0,1,1],[1,0,1],[1,1,1]],dtype=torch.int64)
        for row in possibilities:
            if torch.equal(node,row):
                return True
        return False

def is_rook(node):
        possibilities = torch.tensor([[0,1,0],[1,1,0],[0,1,1],[1,1,1]],dtype=torch.int64)
        for row in possibilities:
            if torch.equal(node,row):
                return True
        return False

def add_diagonal_structure(data_list):
        """
        Add diagonal edges to the dataset.
        """
        new_data_list = []
        for i,graph in enumerate(data_list):
            # find the king and rook
            black_king = None
            white_king = None
            for j, node in enumerate(graph.x):
                if is_black_king(node):
                    black_king = j
                if is_white_king(node):
                    white_king = j

            # add diagonal edges for black king
            new_edge_index = graph.edge_index
            # left up
            if (black_king - 8 - 1) in range(0,64) and black_king % 8 != 0:
                new_edges = torch.tensor([[black_king,black_king-8-1],[black_king-8-1,black_king]],dtype=torch.int64)
                new_edge_index = torch.concatenate((new_edge_index,new_edges),dim=1)
            # right up
            if (black_king - 8 + 1) in range(0,64) and black_king % 8 != 7:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[black_king],[black_king-8+1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[black_king-8+1],[black_king]],dtype=torch.int64)),dim=1)
            # left down
            if (black_king + 8 -1 ) in range(0,64) and black_king % 8 != 0:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[black_king],[black_king+8-1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[black_king+8-1],[black_king]],dtype=torch.int64)),dim=1)
            # right down
            if (black_king + 8 + 1) in range(0,64) and black_king % 8 != 7:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[black_king],[black_king + 8 + 1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[black_king + 8 + 1],[black_king]],dtype=torch.int64)),dim=1)
            
            # add diagonal edges for white king
            # left up
            if (white_king - 8 - 1) in range(0,64) and white_king % 8 != 0:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king],[white_king - 8 - 1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king - 8 - 1],[white_king]],dtype=torch.int64)),dim=1)
            # right up 
            if (white_king - 8 + 1) in range(0,64) and white_king % 8 != 7:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king],[white_king - 8 + 1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king - 8 + 1],[white_king]],dtype=torch.int64)),dim=1)
            # left down
            if (white_king + 8 - 1) in range(0,64) and white_king % 8 != 0:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king],[white_king + 8 -1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king + 8 -1],[white_king]],dtype=torch.int64)),dim=1)
            # right down
            if (white_king + 8 + 1) in range(0,64) and white_king % 8 != 7:
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king],[white_king + 8 + 1]],dtype=torch.int64)),dim=1)
                new_edge_index =torch.concatenate((new_edge_index,torch.tensor([[white_king + 8 + 1],[white_king]],dtype=torch.int64)),dim=1)
            new_graph = Data(x=graph.x,edge_index=new_edge_index,y=graph.y,edge_attr=graph.edge_attr)
            new_data_list.append(new_graph)
        
        return new_data_list
        

def second_representation():
    base = create_base()

    # add diagonal structure
    data_list = add_diagonal_structure(base)
    return data_list
    

def third_representation():
    data_list = []
    for i in range(total_graphs):
        # node features
        # per wking, bking, rook file rank combination
        x = torch.zeros(3, 2,dtype=torch.int)

        # edge indices
        edge_indices = torch.tensor([[0,1,2,0,1,2],[1,0,1,2,2,0]],dtype=torch.int64)

        # fill node attributes
        # first node is the white king
        white_king = white_kings.iloc[i]
        x[0] = torch.tensor([white_king['file'],white_king['rank']],dtype=torch.int64)
        white_rook = white_rooks.iloc[i]
        x[1] = torch.tensor([white_rook['file'],white_rook['rank']],dtype=torch.int64)
        black_king = black_kings.iloc[i]
        x[2] = torch.tensor([black_king['file'],black_king['rank']],dtype=torch.int64)

        # Target y
        y = torch.tensor([0]) if classes.iloc[i]['class'] == 'illegal' else torch.tensor([1])

        # create default edge features
        edge_attr = torch.ones((edge_indices.shape[1],1),dtype=torch.float)
        # create data object
        data = Data(x=x, edge_index=edge_indices, y=y, edge_attr=edge_attr)
        
        # append to list
        data_list.append(data)
    return data_list

def get_rook_neighborhood(i):
    nodes = []
    # node above rook
    node = i + 8
    while node <= 63:
        nodes.append(node)
        node += 8
    # node below rook
    node = i - 8
    while node >= 0:
        nodes.append(node)
        node -= 8
    # node left of rook
    node = i - 1
    while node >= (i//8)*8:
        nodes.append(node)
        node -= 1
    # node right of rook
    node = i + 1
    while node < (i//8)*8 + 8:
        nodes.append(node)
        node += 1
    
    
    return nodes

def get_attack_edges(data):
    edge_indices = []
    for i,row in enumerate(data.x):
        if is_rook(row):
            indices = get_rook_neighborhood(i, edge_indices)
            
def fourth_representation():
    base = second_representation()






data_list = create_base()
torch.save(data_list,'datasets/KRK/FullBoard/raw/datalist.pt')
y_pos = [i for i in data_list if torch.equal(i.y,torch.tensor([1]))]
y_neg = [i for i in data_list if torch.equal(i.y,torch.tensor([0]))]
print("Total legal: ",len(y_pos))
print("Total illegal: ",len(y_neg))

### Test FullBoard representation ###
graph = data_list[0]
print(graph.x.shape)
print(graph.y)
print("amount of examples: ",len(data_list))

data_list_second = second_representation()
torch.save(data_list_second,'datasets/KRK/FullDiag/raw/datalist.pt')
y_pos = [i for i in data_list_second if torch.equal(i.y,torch.tensor([1]))]
y_neg = [i for i in data_list_second if torch.equal(i.y,torch.tensor([0]))]
print("Total illegal: ",len(y_pos))
print("Total legal: ",len(y_neg))

data_list_third = third_representation()
torch.save(data_list_third,'datasets/KRK/Simple/raw/datalist.pt')
y_pos = [i for i in data_list_third if torch.equal(i.y,torch.tensor([1]))]
y_neg = [i for i in data_list_third if torch.equal(i.y,torch.tensor([0]))]
print("Total illegal: ",len(y_pos))
print("Total legal: ",len(y_neg))





