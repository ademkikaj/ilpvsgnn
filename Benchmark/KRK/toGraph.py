import pandas as pd
import os
import glob
import torch_geometric
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.data import HeteroData
import re
from ..GraphConversion import GraphConversion

class toGraph(GraphConversion):
    
    # CHECKED 
    def node_only(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # assumption that every table has an problemId column
            num_nodes = 3
            

            # create node feature array
            num_node_features = 6
            node_features = np.zeros((num_nodes, num_node_features))

            # create edge indices array : fully bidirectionally connected graph
            amount_of_edges = (num_nodes * (num_nodes - 1))
            edge_index = np.zeros((2,amount_of_edges))

            # create edge feature array
            edge_features = np.ones((amount_of_edges,1))

            # create node features
            total_index = 0
            for key in current_kb.keys():
                for index,row in current_kb[key].iterrows():
                    if key == "krk":
                        node_features[total_index][0] = row["white_king_file"]
                        node_features[total_index][1] = row["white_king_rank"]
                        total_index += 1
                        node_features[total_index][2] = row["white_rook_file"]
                        node_features[total_index][3] = row["white_rook_rank"]
                        total_index += 1
                        node_features[total_index][4] = row["black_king_file"]
                        node_features[total_index][5] = row["black_king_rank"]
                        total_index += 1
            
            # fill in the edge indices
            index = 0
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index[0][index] = i
                        edge_index[1][index] = j
                        index += 1
            
            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = torch.tensor(edge_index, dtype=torch.int64),
                edge_attr = torch.tensor(edge_features, dtype=torch.float),
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list

    def node_edge(self):
        # Node predicates -> determine what nodes are made and their respective node features
        # Edge predicates -> determine what nodes are connected with each other, possibly determines edge features

        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # create node features
            num_of_node_features = 6
            num_nodes = 3
            
            node_features = np.zeros((num_nodes, num_of_node_features))

            node_mapping = {}
            index = 0
            for _,row in current_kb["krk"].iterrows():
                
                node_features[index][0] = row["white_king_file"]
                node_features[index][1] = row["white_king_rank"]
                index += 1
                node_features[index][2] = row["white_rook_file"]
                node_features[index][3] = row["white_rook_rank"]
                index += 1
                node_features[index][4] = row["black_king_file"]
                node_features[index][5] = row["black_king_rank"]
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            # fill in the edges and edge features
            self.add_edges(edge_indices,0,1)
            # calculate the distance 
            distance = abs(current_kb["krk"]["white_king_file"].iloc[0] - current_kb["krk"]["white_rook_file"].iloc[0]) + abs(current_kb["krk"]["white_king_rank"].iloc[0] - current_kb["krk"]["white_rook_rank"].iloc[0])
            edge_feat = [0,0,0]
            if current_kb["krk"]["white_king_file"].iloc[0] == current_kb["krk"]["white_rook_file"].iloc[0]:
                edge_feat[0] = 1
            if current_kb["krk"]["white_king_rank"].iloc[0] == current_kb["krk"]["white_rook_rank"].iloc[0]:
                edge_feat[1] = 1
            edge_feat[2] = distance
            #edge_feat = [0]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            self.add_edges(edge_indices,1,2)
            distance = abs(current_kb["krk"]["white_rook_file"].iloc[0] - current_kb["krk"]["black_king_file"].iloc[0]) + abs(current_kb["krk"]["white_rook_rank"].iloc[0] - current_kb["krk"]["black_king_rank"].iloc[0])
            edge_feat = [0,0,0]
            if current_kb["krk"]["white_rook_file"].iloc[0] == current_kb["krk"]["black_king_file"].iloc[0]:
                edge_feat[0] = 1
            if current_kb["krk"]["white_rook_rank"].iloc[0] == current_kb["krk"]["black_king_rank"].iloc[0]:
                edge_feat[1] = 1
            edge_feat[2] = distance
            #edge_feat = [0]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            self.add_edges(edge_indices,0,2)
            distance = abs(current_kb["krk"]["white_king_file"].iloc[0] - current_kb["krk"]["black_king_file"].iloc[0]) + abs(current_kb["krk"]["white_king_rank"].iloc[0] - current_kb["krk"]["black_king_rank"].iloc[0])
            edge_feat = [0,0,0]
            if current_kb["krk"]["white_king_file"].iloc[0] == current_kb["krk"]["black_king_file"].iloc[0]:
                edge_feat[0] = 1
            if current_kb["krk"]["white_king_rank"].iloc[0] == current_kb["krk"]["black_king_rank"].iloc[0]:
                edge_feat[1] = 1
            edge_feat[2] = distance
            #edge_feat = [0]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            

            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            # normalise the edge_features via the columns
            col_sums = edge_features.sum(axis=0)
            edge_features = edge_features / col_sums

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = edge_features,
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list
    
    def edge_based(self):
        
        data_list = []
        for graph_index in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]

            current_kb = self.get_current_objects(graph_id)

            # 3 for the external predicates and 1 for instances
            num_node_features = 6 + 3 # 1 for the atom fact and 1 for the instances

            # external and internal predicates
            num_edge_features = 4 # 1 for external and 1 for internal relations and then 2 more same_rank or same_file
            
            num_nodes = 6
            # create the node features 
            node_features = np.zeros((num_nodes, num_node_features))


            node_features[0][0] = 1
            node_features[1][1] = 1
            node_features[2][2] = 1


            index = 3
            for key in current_kb.keys():
                for _,row in current_kb[key].iterrows():
                    if key == "krk":
                        node_features[index][3] = row["white_king_file"]
                        node_features[index][4] = row["white_king_rank"]
                        index += 1
                        node_features[index][5] = row["white_rook_file"]
                        node_features[index][6] = row["white_rook_rank"]
                        index += 1
                        node_features[index][7] = row["black_king_file"]
                        node_features[index][8] = row["black_king_rank"]
                        index += 1

            edge_indices = [[],[]]
            edge_features = []
            # add the external relations
            self.add_edges(edge_indices,0,3)
            edge_features.append([1,0,0,0,0])
            edge_features.append([1,0,0,0,0])
            # edge_features.append([1,0])
            # edge_features.append([1,0])
            self.add_edges(edge_indices,1,4)
            edge_features.append([1,0,0,0,0])
            edge_features.append([1,0,0,0,0])
            # edge_features.append([1,0])
            # edge_features.append([1,0])
            self.add_edges(edge_indices,2,5)
            #self.add_edges(edge_indices,2,5)





            edge_features.append([1,0,0,0,0])
            edge_features.append([1,0,0,0,0])
            # edge_features.append([1,0])
            # edge_features.append([1,0])

            # add the internal relations
            self.add_edges(edge_indices,3,4)
            edge_feat = [0,1]
            edge_feat = [0,1,0,0,0]
            distance = abs(current_kb["krk"]["white_king_file"].iloc[0] - current_kb["krk"]["white_rook_file"].iloc[0]) + abs(current_kb["krk"]["white_king_rank"].iloc[0] - current_kb["krk"]["white_rook_rank"].iloc[0])
            edge_feat[-1] = distance
            if current_kb["krk"]["white_king_file"].iloc[0] == current_kb["krk"]["white_rook_file"].iloc[0]:
                edge_feat[2] = 1
            if current_kb["krk"]["white_king_rank"].iloc[0] == current_kb["krk"]["white_rook_rank"].iloc[0]:
                edge_feat[3] = 1
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            self.add_edges(edge_indices,4,5)
            edge_feat = [0,1]
            edge_feat = [0,1,0,0,0]
            distance = abs(current_kb["krk"]["white_rook_file"].iloc[0] - current_kb["krk"]["black_king_file"].iloc[0]) + abs(current_kb["krk"]["white_rook_rank"].iloc[0] - current_kb["krk"]["black_king_rank"].iloc[0])
            edge_feat[-1] = distance
            if current_kb["krk"]["white_rook_file"].iloc[0] == current_kb["krk"]["black_king_file"].iloc[0]:
                edge_feat[2] = 1
            if current_kb["krk"]["white_rook_rank"].iloc[0] == current_kb["krk"]["black_king_rank"].iloc[0]:
                edge_feat[3] = 1
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            self.add_edges(edge_indices,3,5)
            edge_feat = [0,1]
            edge_feat = [0,1,0,0,0]
            distance = abs(current_kb["krk"]["white_king_file"].iloc[0] - current_kb["krk"]["black_king_file"].iloc[0]) + abs(current_kb["krk"]["white_king_rank"].iloc[0] - current_kb["krk"]["black_king_rank"].iloc[0])
            edge_feat[-1] = distance
            if current_kb["krk"]["white_king_file"].iloc[0] == current_kb["krk"]["black_king_file"].iloc[0]:
                edge_feat[2] = 1
            if current_kb["krk"]["white_king_rank"].iloc[0] == current_kb["krk"]["black_king_rank"].iloc[0]:
                edge_feat[3] = 1
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            edge_features = torch.tensor(edge_features, dtype=torch.float)
            col_sums = edge_features.sum(axis=0)
            edge_features = edge_features / col_sums
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attr = edge_features
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label))

        return data_list

    def Klog(self):
        data_list = []

        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)

            # 3 for the external predicates and 1 for instances
            num_node_features = 6 + 2 # 1 for edge predicates
            
            row = current_kb["krk"].iloc[0]
            num_extra_nodes = 0
            num_extra_nodes += int(row["white_king_file"] == row["white_rook_file"])
            num_extra_nodes += int(row["white_king_file"] == row["black_king_file"])
            num_extra_nodes += int(row["white_rook_file"] == row["black_king_file"])
            num_extra_nodes += int(row["white_king_rank"] == row["white_rook_rank"])
            num_extra_nodes += int(row["white_king_rank"] == row["black_king_rank"])
            num_extra_nodes += int(row["white_rook_rank"] == row["black_king_rank"])

            num_nodes = 3 + num_extra_nodes + 3 

            node_features = np.zeros((num_nodes, num_node_features))

            node_features[0][0] = row["white_king_file"]
            node_features[0][1] = row["white_king_rank"]
            node_features[1][2] = row["white_rook_file"]
            node_features[1][3] = row["white_rook_rank"]
            node_features[2][4] = row["black_king_file"]
            node_features[2][5] = row["black_king_rank"]

            # add the extra nodes and edges
            edge_indices = [[],[]]
            edge_features = []
            index = 3
            if row["white_king_file"] == row["white_rook_file"]:
                node_features[index][6] = 1
                self.add_edges(edge_indices,0,index)
                self.add_edges(edge_indices,1,index)
                edge_feat = [0]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                index += 1
            if row["white_king_rank"] == row["white_rook_rank"]:
                node_features[index][7] = 1
                self.add_edges(edge_indices,0,index)
                self.add_edges(edge_indices,1,index)
                edge_feat = [0]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                index += 1
            if row["white_king_file"] == row["black_king_file"]:
                node_features[index][6] = 1
                self.add_edges(edge_indices,0,index)
                self.add_edges(edge_indices,2,index)
                edge_feat = [0]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                index += 1
            if row["white_king_rank"] == row["black_king_rank"]:
                node_features[index][7] = 1
                self.add_edges(edge_indices,0,index)
                self.add_edges(edge_indices,2,index)
                edge_feat = [0]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                index += 1
            if row["white_rook_file"] == row["black_king_file"]:
                node_features[index][6] = 1
                self.add_edges(edge_indices,1,index)
                self.add_edges(edge_indices,2,index)
                edge_feat = [0]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                index += 1
            if row["white_rook_rank"] == row["black_king_rank"]:
                node_features[index][7] = 1
                self.add_edges(edge_indices,1,index)
                self.add_edges(edge_indices,2,index)
                edge_feat = [0]
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
                index += 1

            # add edges between the nodes
            node_features[3][6] = 1
            self.add_edges(edge_indices,0,3)
            self.add_edges(edge_indices,3,1)
            distance = abs(row["white_king_file"] - row["white_rook_file"]) + abs(row["white_king_rank"] - row["white_rook_rank"])
            edge_feat = [distance]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            node_features[4][6] = 1
            self.add_edges(edge_indices,1,4)
            self.add_edges(edge_indices,4,2)
            distance = abs(row["white_rook_file"] - row["black_king_file"]) + abs(row["white_rook_rank"] - row["black_king_rank"])
            edge_feat = [distance]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

            node_features[5][6] = 1
            self.add_edges(edge_indices,0,5)
            self.add_edges(edge_indices,5,2)
            distance = abs(row["white_king_file"] - row["black_king_file"]) + abs(row["white_king_rank"] - row["black_king_rank"])
            edge_feat = [distance]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)
            

            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)
            edge_features = torch.tensor(edge_features, dtype=torch.float)

            col_sums = edge_features.sum(axis=0)
            edge_features = edge_features / col_sums

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=truth_label))
        return data_list

    def FullBoard(self):
        data_list = []
        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.get_current_objects(graph_id)
            current_kb = current_kb["krk"]
            assert len(current_kb) == 1
            # white king, white rook, black king
            # node attributes
            x = torch.zeros(64, 3)
            # edge indices
            total_edges  = (8*7)*4
            edge_indices = torch.zeros((2, total_edges), dtype=torch.int64)

            # fill node attributes
            x_index = int(current_kb['white_king_file']*8 + current_kb['white_king_rank'].iloc[0])
            x[x_index][0] = 1

            x_index = int(current_kb['white_rook_file']*8 + current_kb['white_rook_rank'].iloc[0])
            x[x_index][1] = 1

            x_index = int(current_kb['black_king_file']*8 + current_kb['black_king_rank'].iloc[0])
            x[x_index][2] = 1

            #assert white_rook['id'] == white_king['id'] == black_king['id']
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
            
            
            # create default edge features
            edge_attr = torch.ones((edge_indices.shape[1],1),dtype=torch.float)
            # create data object
            data = Data(x=x, edge_index=edge_indices, y=self.y[graph_index], edge_attr=edge_attr)
            
            # append to list
            data_list.append(data)
        return data_list

    def FullDiag(self):
        base = self.FullBoard()

        data_list = self.add_diagonal_structure(base)
        return data_list

    def add_diagonal_structure(self,data_list):
        """
        Add diagonal edges to the dataset.
        """
        new_data_list = []
        for i,graph in enumerate(data_list):
            # find the king and rook
            black_king = None
            white_king = None
            for j, node in enumerate(graph.x):
                if self.is_black_king(node):
                    black_king = j
                if self.is_white_king(node):
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
            edge_attr = torch.ones((new_edge_index.shape[1],1),dtype=torch.float)
            new_graph = Data(x=graph.x,edge_index=new_edge_index,y=graph.y,edge_attr=edge_attr)
            new_data_list.append(new_graph)
        
        return new_data_list
    
    def is_white_king(self,node):
        possibilities = torch.tensor([[1,0,0],[1,1,0],[1,0,1],[1,1,1]],dtype=torch.int64)
        for row in possibilities:
            if torch.equal(node,row):
                return True
        return False

    def is_black_king(self,node):
            possibilities = torch.tensor([[0,0,1],[0,1,1],[1,0,1],[1,1,1]],dtype=torch.int64)
            for row in possibilities:
                if torch.equal(node,row):
                    return True
            return False

    def is_rook(self,node):
            possibilities = torch.tensor([[0,1,0],[1,1,0],[0,1,1],[1,1,1]],dtype=torch.int64)
            for row in possibilities:
                if torch.equal(node,row):
                    return True
            return False
