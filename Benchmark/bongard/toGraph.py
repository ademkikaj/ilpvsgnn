import pandas as pd
import os
import glob
import torch_geometric
import numpy as np
import torch
#from BongardDataset import BongardDataset
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.data import HeteroData
import re

class toGraph:
    def __init__(self, relational_path,dataset_name,dataset_problem_key,target):
        self.relational_path = relational_path
        self.dataset_name = dataset_name
        self.problem_key = dataset_problem_key
        self.target = target
        self.kb = self.collect_relational()
        self.num_of_graphs = len(self.kb[self.dataset_name][self.problem_key].unique())
        self.num_node_features = 3
        self.y = self.create_y(self.kb)


    def collect_relational(self):
        kb = {}
        predicates = []
        all_files = glob.glob(os.path.join(self.relational_path, "*.csv"))
        for filename in all_files:
            df = pd.read_csv(filename)
            relation = filename.split('/')[-1].split('.')[0]
            predicates.append(relation)
            kb[relation] = df
        # sort all the dataframes by the problemId
        for key in kb.keys():
            kb[key] = kb[key].sort_values(by=[self.problem_key])
        return kb
    
    def create_y(self, kb):
        y = kb[self.dataset_name][self.target].to_numpy()
        y[y == "neg"] = 0
        y[y == " neg"] = 0
        y[y == "pos"] = 1
        y[y == " pos"] = 1
        return y
    
    def add_edges(self,edge_indices,node_index1,node_index2,edge_index):
        edge_indices[0].append(node_index1)
        edge_indices[1].append(node_index2)
        edge_index += 1
        edge_indices[0].append(node_index2)
        edge_indices[1].append(node_index1)
        edge_index += 1
    
    def collect_problemId_objects(self,index):
        # df = self.kb["bongard"]
        # graph_id = df[df["problemId"] == index]
        graph_id = index
        current_kb = {}
        current_kb["triangle"] = self.kb["triangle"][self.kb["triangle"][self.problem_key] == graph_id]
        current_kb["square"] = self.kb["square"][self.kb["square"][self.problem_key] == graph_id]
        current_kb["circle"] = self.kb["circle"][self.kb["circle"][self.problem_key] == graph_id]
        current_kb["in"] = self.kb["in"][self.kb["in"][self.problem_key] == graph_id]
        new_shapes = ["shape1","shape2","shape3","shape4","shape5"]
        for shape in new_shapes:
            if shape in self.kb.keys():
                current_kb[shape] = self.kb[shape][self.kb[shape][self.problem_key] == graph_id]
        
        return current_kb
        
    def NoFrameRepresentation(self):
        data_list = []
        for i in range(self.num_of_graphs):
            # reset params
            node_index,edge_index,edge_indices = 0,0,[[],[]]

            triangles,squares,circles,ins = self.collect_problemId_objects(i)

            objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.unique(objectIds) # returns sorted unique elements of an array

            node_features = np.zeros((len(objectIds), self.num_node_features))
            start_object_index = node_index

            # a node is connected to the background node if it never appears as objectId1
            for objectId in objectIds:
                # add the node to node features
                if objectId in squares["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([1, 0, 0])
                elif objectId in circles["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([0, 1, 0])
                elif objectId in triangles["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([0, 0, 1])
                    
                # check if there is an object inside current objectId
                if objectId in ins["objectId2"].to_numpy():
                    # connect the node to the object inside, bidirectional
                    current_ins = ins[ins["objectId2"] == objectId]
                    for childId in current_ins["objectId1"].to_numpy():
                        child_index = np.where(objectIds == childId)[0][0] + start_object_index
                        self.add_edges(edge_indices,node_index,child_index,edge_index)
                node_index += 1
            data_list += [
                Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.int64),
                    y=torch.tensor(self.y[i], dtype=torch.int64),
                    edge_attr=torch.ones(len(edge_indices[0]),1)
                )
            ]

        return data_list

    def NodeOnlyRepresentation(self):
        # connect every node to the frame node
        data_list = []
        for i in range(self.num_of_graphs):
            # reset params
            node_index,edge_index,edge_indices = 0,0,[[],[]]

            triangles,squares,circles,ins = self.collect_problemId_objects(i)
            objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.unique(objectIds) # returns sorted unique elements of an array

            # init the node features for the objects, for every object there is a node and a there is a global background node
            node_features = np.zeros((len(objectIds)+1, self.num_node_features))

            # add the background node
            background_index = node_index
            node_features[node_index][:] = np.array([0, 0, 0])
            node_index += 1

            start_object_index = background_index + 1

            # a node is connected to the background node if it never appears as objectId1
            for objectId in objectIds:
                # add the node to node features
                if objectId in squares["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([1, 0, 0])
                elif objectId in circles["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([0, 1, 0])
                elif objectId in triangles["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([0, 0, 1])

                # have to connect to background node to all other nodes 
                self.add_edges(edge_indices,background_index,node_index,edge_index)
                    
                # check if there is an object inside current objectId
                if objectId in ins["objectId2"].to_numpy():
                    # connect the node to the object inside, bidirectional
                    current_ins = ins[ins["objectId2"] == objectId]
                    for childId in current_ins["objectId1"].to_numpy():
                        child_index = np.where(objectIds == childId)[0][0] + start_object_index
                        self.add_edges(edge_indices,node_index,child_index,edge_index)
                
                node_index += 1
            data_list += [
                Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.int64),
                    y=torch.tensor(self.y[i], dtype=torch.int64),
                    edge_attr=torch.ones(len(edge_indices[0]),1)
                )
            ]
            
        return data_list

    def EdgeNodeRepresentation(self):
        data_list = []
        for i in range(self.num_of_graphs):
            # reset params
            node_index = 0
            edge_index = 0
            edge_indices = [[],[]]

            triangles,squares,circles,ins = self.collect_problemId_objects(i)


            objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.unique(objectIds)

            # create node feature array
            num_of_nodes = len(objectIds)
            node_features = np.zeros((num_of_nodes, self.num_node_features))
            
            # create edge indices array
            amount_of_edges = sum(range(num_of_nodes))*2
            edge_indices = np.zeros((2,amount_of_edges))

            # create edge feature array
            num_edge_features = 1        # edge feature is 1 for the edge (vi,vj) if vi is inside vj
            edge_features = np.zeros((amount_of_edges, num_edge_features))

            for obj in objectIds:
                # add the node to node features
                if obj in squares["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([1, 0, 0])
                elif obj in circles["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([0, 1, 0])
                elif obj in triangles["objectId"].to_numpy():
                    node_features[node_index][:] = np.array([0, 0, 1])
                
                # add edges to all other nodes
                other_nodes = list(range(num_of_nodes))
                other_nodes.remove(node_index)
                for j in other_nodes:
                    edge_indices[0][edge_index] = node_index
                    edge_indices[1][edge_index] = j
                    obj2 = objectIds[j]
                    if ins[(ins["objectId1"] == obj) & (ins["objectId2"] == obj2)].empty:
                        edge_features[edge_index] = 0
                    else:
                        edge_features[edge_index] = 1
                    edge_index += 1

                node_index += 1
            
            x = torch.tensor(node_features, dtype=torch.int64)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attr = torch.tensor(edge_features, dtype=torch.int64)
            truth_label = torch.tensor(self.y[i],dtype=torch.int64)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label)
            data_list += [data]
        return data_list
        
    def KlogRepresentation(self):
        frame         = [0,0,0,0,0]
        square        = [1,0,0,0,0]
        circle        = [0,1,0,0,0]
        triangle      = [0,0,1,0,0]
        in_node       = [0,0,0,1,0]
        part_of_node  = [0,0,0,0,1]

        data_list = []
        for i in range(self.num_of_graphs):
            # reset params
            node_index = 0
            edge_index = 0
            index_dict = {}
            
            triangles,squares,circles,ins = self.collect_problemId_objects(i)

            objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.unique(objectIds)

            # start with creating a frame node
            index_dict["frame"] = node_index
            node_index += 1

            # Give all object ids a part of node
            for obj in objectIds:
                index_dict[obj + '_part_of'] = node_index
                node_index += 1
                index_dict[obj] = node_index
                node_index += 1

            # loop over all objects that arent inside another object
            
            # loop over all in relations
            in_relations = []
            for j in range(len(ins)):
                obj1 = ins.iloc[j]["objectId1"]
                obj2 = ins.iloc[j]["objectId2"]
                in_relations.append((obj1,obj2))
                index_dict[obj1 + obj2 + '_in'] = node_index
                node_index += 1
            # create x
            x = torch.zeros((node_index,5), dtype=torch.float)
            for key,value in index_dict.items():
                if key == "frame":
                    x[value] = torch.tensor(frame)
                elif key[-3:] == "_of":
                    x[value] = torch.tensor(part_of_node)
                elif key[-3:] == "_in":
                    x[value] = torch.tensor(in_node)
                else:
                    if key in squares["objectId"].to_numpy():
                        x[value] = torch.tensor(square)
                    elif key in circles["objectId"].to_numpy():
                        x[value] = torch.tensor(circle)
                    elif key in triangles["objectId"].to_numpy():
                        x[value] = torch.tensor(triangle)
            
            # create edge_index
            edge_index = [[],[]]
            
            for key,value in index_dict.items():
                
                if key[-3:] == "_of":
                    # create the edges between the frame node and the part of nodes
                    edge_index[0].append(index_dict["frame"])
                    edge_index[1].append(value)
                    edge_index[0].append(value)
                    edge_index[1].append(index_dict["frame"])
                    # create the edges between the part of nodes and the shape nodes
                    match = re.search(r'(o.*?)_part_of+', key)
                    obj = match.group(1)

                    edge_index[0].append(value)
                    edge_index[1].append(index_dict[obj])
                    edge_index[0].append(index_dict[obj])
                    edge_index[1].append(value)


                # create the edges between the in nodes and the shape nodes
                elif key[-3:] == "_in":
                    match = re.match(r'(o.*?)o(.*?)(?:_in)?$', key)
                    obj1 = match.group(1)
                    obj2 = 'o' + match.group(2)

                    edge_index[0].append(index_dict[obj1])
                    edge_index[1].append(value)
                    edge_index[0].append(value)
                    edge_index[1].append(index_dict[obj1])

                    edge_index[0].append(index_dict[obj2])
                    edge_index[1].append(value)
                    edge_index[0].append(value)
                    edge_index[1].append(index_dict[obj2])

            edge_index = torch.tensor(edge_index, dtype=torch.long)
            # create edge_attr
            edge_attr = torch.ones((len(edge_index[0]),1), dtype=torch.float)

            # create y
            truth_label = torch.tensor(self.y[i], dtype=torch.long)

            # create graph
            data_list += [Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label)]
        return data_list
    
    # CHECKED 
    def node_only(self):
        # Fully connected graph with node features only

        problem_key = "problemId"
        shapes = ["triangle", "square", "circle","shape1","shape2","shape3","shape4","shape5"]
        shapes = ["triangle", "square", "circle"]
        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.collect_problemId_objects(graph_id)
            
            #objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.concatenate([current_kb[shape]["objectId"].to_numpy() for shape in shapes if shape in current_kb.keys()])
            objectIds = np.unique(objectIds)

            num_of_node_features = len([ 1 for shape in shapes if shape in current_kb.keys()]) + 1

            # create node feature array
            num_nodes = len(objectIds) + len(current_kb["in"])
            node_features = np.zeros((num_nodes, num_of_node_features))

            # create edge indices array : fully bidirectionally connected graph
            amount_of_edges = (num_nodes * (num_nodes - 1))
            edge_index = np.zeros((2,amount_of_edges))

            # create edge feature array
            edge_features = np.ones((amount_of_edges,1))

            # create node features
            for index,obj in enumerate(objectIds):
                for shape in current_kb.keys():
                    if shape != "in":
                        if obj in current_kb[shape]["objectId"].to_numpy():
                            current_shape = shape
                node_features[index][shapes.index(current_shape)] = 1
                # if obj in ["objectId"].to_numpy():
                #     node_features[index][0] = 1.0
                # if obj in squares["objectId"].to_numpy():
                #     node_features[index][1] = 1.0
                # if obj in circles["objectId"].to_numpy():
                #     node_features[index][2] = 1.0
            
            # iterate over rows in ins
            for k in range(len(current_kb["in"])):
                node_features[k+len(objectIds)][3] = 1
            
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

        problem_key = "problemId"

        node_predicates = ["square", "circle", "triangle"]
        edge_predicates = ["in"]
        shapes = ["triangle", "square", "circle","shape1","shape2","shape3","shape4","shape5"]

        data_list = []
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.collect_problemId_objects(graph_id)

            #objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.concatenate([current_kb[shape]["objectId"].to_numpy() for shape in shapes if shape in current_kb.keys()])
            objectIds = np.unique(objectIds)
            # if the name of variable is the same as the name of the predicate, then the variable is the predicate

            # create node features
            num_of_node_features = len([ 1 for shape in shapes if shape in current_kb.keys()])
            num_nodes = len(objectIds)

            node_features = np.zeros((num_nodes, num_of_node_features))

            for index,obj in enumerate(objectIds):
                # find the shape in which object is 
                for shape in current_kb.keys():
                    if shape != "in":
                        if obj in current_kb[shape]["objectId"].to_numpy():
                            current_shape = shape
                node_features[index][shapes.index(current_shape)] = 1
                # if obj in squares["objectId"].to_numpy():
                #     node_features[index][0] = 1
                # if obj in circles["objectId"].to_numpy():
                #     node_features[index][1] = 1
                # if obj in triangles["objectId"].to_numpy():
                #     node_features[index][2] = 1
            
            # create edge indices
            edge_indices = [[],[]]
            edge_features = []
            for index,row in current_kb["in"].iterrows():
                obj1 = row["objectId1"]
                obj2 = row["objectId2"]
                index1 = np.where(objectIds == obj1)[0][0]
                index2 = np.where(objectIds == obj2)[0][0]
                edge_indices[0].append(index1)
                edge_indices[1].append(index2)
                edge_indices[0].append(index2)
                edge_indices[1].append(index1)

                # get the shape of objectId1
                # if obj1 in triangles["objectId"].to_numpy():
                #     # if obj2 in triangles["objectId"].to_numpy():
                #     #     edge_features.append([1])
                #     #     edge_features.append([1])
                #     if obj2 in circles["objectId"].to_numpy():
                #         edge_features.append([1])
                #         edge_features.append([1])
                #     else:
                #         edge_features.append([0])
                #         edge_features.append([0])
                # else:
                #     edge_features.append([0])
                #     edge_features.append([0])

            edge_index = torch.tensor(edge_indices, dtype=torch.int64)

            # create edge features
            edge_features = torch.ones((len(edge_indices[0]),1), dtype=torch.float)
            #edge_features = torch.tensor(edge_features, dtype=torch.float)

            data_list.append(Data(
                x = torch.tensor(node_features, dtype=torch.float),
                edge_index = edge_index,
                edge_attr = edge_features,
                y = torch.tensor(self.y[graph_index], dtype=torch.int64)
            ))
        return data_list
    
    def edge_based(self):
        problem_key = "problemId"
        data_list = []
        shapes = ["triangle", "square", "circle","shape1","shape2","shape3","shape4","shape5"]
        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.collect_problemId_objects(graph_id)
            objectIds = np.concatenate([current_kb[shape]["objectId"].to_numpy() for shape in shapes if shape in current_kb.keys()])
            #objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.unique(objectIds)

            # 3 for the external predicates and 1 for instances
            num_node_features = len([ 1 for shape in shapes if shape in current_kb.keys()]) + 1
            num_shapes = len([ 1 for shape in shapes if shape in current_kb.keys()])
            # external and internal predicates
            num_edge_features = 2

            # create the node features 
            node_features = np.zeros((len(objectIds) + num_shapes, num_node_features))

            for i in range(num_node_features-1):
                node_features[i][0] = 1
            for i in range(num_node_features,len(objectIds) + num_shapes):
                node_features[i][-1] = 1

            # create the edges
            object_indices = {shape:i for i,shape in enumerate(shapes)}
            edge_indices = [[],[]]
            edge_features = []

            for index,obj in enumerate(objectIds):
                for shape in [shap for shap in shapes if shap in current_kb.keys()]:
                    if obj in current_kb[shape]["objectId"].to_numpy():
                        edge_indices[0].append(object_indices[shape])
                        edge_indices[1].append(index + len([shap for shap in shapes if shap in current_kb.keys()]))
                        edge_features.append([1,0])
                        edge_indices[0].append(index + len([shap for shap in shapes if shap in current_kb.keys()]))
                        edge_indices[1].append(object_indices[shape])
                        edge_features.append([1,0])

                # if obj in squares["objectId"].to_numpy():
                #     edge_indices[0].append(0)
                #     edge_indices[1].append(index + 3)
                #     #edge_features.append([1,0])
                #     edge_features.append([1,0,0])
                #     edge_indices[0].append(index + 3)
                #     edge_indices[1].append(0)
                #     #edge_features.append([1,0])
                #     edge_features.append([1,0,0])
                # if obj in circles["objectId"].to_numpy():
                #     edge_indices[0].append(1)
                #     edge_indices[1].append(index + 3)
                #     #edge_features.append([1,0])
                #     edge_features.append([1,0,0])
                #     edge_indices[0].append(index + 3)
                #     edge_indices[1].append(1)
                #     #edge_features.append([1,0])
                #     edge_features.append([1,0,0])
                # if obj in triangles["objectId"].to_numpy():
                #     edge_indices[0].append(2)
                #     edge_indices[1].append(index + 3)
                #     #edge_features.append([1,0])
                #     edge_features.append([1,0,0])
                #     edge_indices[0].append(index + 3)
                #     edge_indices[1].append(2)
                #     #edge_features.append([1,0])
                #     edge_features.append([1,0,0])
            
            for index,row in current_kb["in"].iterrows():
                obj1 = row["objectId1"]
                obj2 = row["objectId2"]
                index1 = np.where(objectIds == obj1)[0][0] + len([shap for shap in shapes if shap in current_kb.keys()])
                index2 = np.where(objectIds == obj2)[0][0] + len([shap for shap in shapes if shap in current_kb.keys()])
                edge_indices[0].append(index1)
                edge_indices[1].append(index2)
                edge_features.append([0,1])
                edge_indices[0].append(index2)
                edge_indices[1].append(index1)
                edge_features.append([0,1])
                # if obj1 in triangles["objectId"].to_numpy():
                #     # if obj2 in triangles["objectId"].to_numpy():
                #     #     edge_features.append([0,1,1])
                #     #     edge_features.append([0,1,1])
                #     if obj2 in circles["objectId"].to_numpy():
                #         edge_features.append([0,1,1])
                #         edge_features.append([0,1,1])
                #     else:
                #         edge_features.append([0,1,0])
                #         edge_features.append([0,1,0])
                # else:
                #     edge_features.append([0,1,0])
                #     edge_features.append([0,1,0])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label))

        return data_list
    
    def Klog(self):

        data_list = []
        shapes = ["triangle", "square", "circle","shape1","shape2","shape3","shape4","shape5"]
        for graph_index in range(self.num_of_graphs):
            
            graph_id = self.kb[self.dataset_name][self.problem_key][graph_index]
            current_kb = self.collect_problemId_objects(graph_id)
            

            #objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
            objectIds = np.concatenate([current_kb[shape]["objectId"].to_numpy() for shape in shapes if shape in current_kb.keys()])
            objectIds = np.unique(objectIds)

            current_shapes = [shape for shape in shapes if shape in current_kb.keys()]
            # create node features
            node_features = np.zeros((len(objectIds) + len(current_kb["in"]), len(current_shapes)+1))

            # create edge indices
            edge_indices = [[],[]]
            edge_features = []

            for index,obj in enumerate(objectIds):
                # if obj in squares["objectId"].to_numpy():
                #     node_features[index][0] = 1
                # if obj in circles["objectId"].to_numpy():
                #     node_features[index][1] = 1
                # if obj in triangles["objectId"].to_numpy():
                #     node_features[index][2] = 1

                # find the shape in which object is 
                for shape in current_kb.keys():
                    if shape != "in":
                        if obj in current_kb[shape]["objectId"].to_numpy():
                            current_shape = shape
                node_features[index][current_shapes.index(current_shape)] = 1
            
            for i in range(len(current_kb["in"])):
                row = current_kb["in"].iloc[i]
                in_index = i + len(objectIds)
                obj1 = row["objectId1"]
                obj2 = row["objectId2"]
                index1 = np.where(objectIds == obj1)[0][0] 
                index2 = np.where(objectIds == obj2)[0][0] 
                # add in node
                node_features[in_index][-1] = 1
                # add 4 edges from the in node to the objects
                edge_indices[0].append(index1)
                edge_indices[1].append(in_index)
                edge_indices[0].append(in_index)
                edge_indices[1].append(index1)

                edge_indices[0].append(index2)
                edge_indices[1].append(in_index)
                edge_indices[0].append(in_index)
                edge_indices[1].append(index2)

                # if obj1 in triangles["objectId"].to_numpy():
                #     # if obj2 in triangles["objectId"].to_numpy():
                #     #     edge_features.append([1])
                #     #     edge_features.append([1])
                #     #     edge_features.append([1])
                #     #     edge_features.append([1])
                #     if obj2 in circles["objectId"].to_numpy():
                #         edge_features.append([1])
                #         edge_features.append([1])
                #         edge_features.append([1])
                #         edge_features.append([1])
                #     else:
                #         edge_features.append([0])
                #         edge_features.append([0])
                #         edge_features.append([0])
                #         edge_features.append([0])
                # else:
                #     edge_features.append([0])
                #     edge_features.append([0])
                #     edge_features.append([0])
                #     edge_features.append([0])
            
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.int64)
            #edge_attr = torch.ones((len(edge_indices[0]),1), dtype=torch.float)
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            truth_label = torch.tensor(self.y[graph_index], dtype=torch.int64)
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=truth_label))

        return data_list
        

            




            



# ### turn this relational data into graph data

# ## 1. Have a starting background node and connect these to the most outside nodes

# def get_num_of_nodes(kb):
#     # node for every shape
#     num_of_nodes = len(kb["triangle"])
#     num_of_nodes += len(kb["square"])
#     num_of_nodes += len(kb["circle"])
#     # Background node per graph
#     num_of_nodes += num_of_graphs
#     return num_of_nodes

# # Node features
# # Have as possible node features [0 -> Background, 1 -> Square, 2 -> Circle, 3 -> Triangle_UP, 4 -> Triangle_DOWN]
# num_of_node_features = 4




# def to_node_value_encoding(datalist):
#         """
#         Convert the dataset from one-hot-encoding to node value encoding.
#         The structure change of the dataset must be performed first.
#         """
#         max_value = data_list[0].x.shape[1]
#         value_ranges = range(1, max_value + 1)
#         new_data_list = []
#         for graph in data_list:
#             new_x = torch.zeros((graph.x.shape[0], 1),dtype=torch.int64)
#             for i,row in enumerate(graph.x):
#                 # check if torch tensor is all zeros
#                 if torch.all(row == 0):
#                     new_x[i] = 0
#                 else:
#                     new_x[i] = value_ranges[row.argmax()]
#             new_graph = Data(x=new_x, edge_index=graph.edge_index, y=graph.y, edge_attr=graph.edge_attr)
#             new_data_list.append(new_graph)
#         return new_data_list


# def create_datalist(kb):
#     data_list = []
#     for i in range(num_of_graphs):
#         # reset params
#         node_index = 0
#         edge_index = 0
#         edge_indices = [[],[]]

#         graph_id = kb["bongard"]["problemId"][i]
#         triangles = kb["triangle"][kb["triangle"]["problemId"] == graph_id]
#         squares = kb["square"][kb["square"]["problemId"] == graph_id]
#         circles = kb["circle"][kb["circle"]["problemId"] == graph_id]
#         configs = kb["config"][kb["config"]["problemId"] == graph_id]
#         # triangles up: traingles whith objectId in configs and orient "up"
#         triangles_up = triangles[triangles["objectId"].isin(configs[configs["orient"] == "up"]["objectId"])]
#         # triangles down: traingles whith objectId in configs and orient "down"
#         triangles_down = triangles[triangles["objectId"].isin(configs[configs["orient"] == "down"]["objectId"])]
#         ins = kb["in"][kb["in"]["problemId"] == graph_id]

#         objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
#         objectIds = np.unique(objectIds) # returns sorted unique elements of an array

#         # init the node features for the objects, for every object there is a node and a there is a global background node
#         node_features = np.zeros((len(objectIds)+1, num_of_node_features))

#         # add the background node
#         background_index = node_index
#         node_features[node_index][:] = np.array([0, 0, 0, 0])
#         node_index += 1

#         start_object_index = background_index + 1

#         # a node is connected to the background node if it never appears as objectId1
#         for objectId in objectIds:
#             # add the node to node features
#             if objectId in squares["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([1, 0, 0, 0])
#             elif objectId in circles["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0, 1, 0, 0])
#             elif objectId in triangles_up["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0, 0, 1, 0])
#             elif objectId in triangles_down["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0, 0, 0, 1])

#             # have to connect to background node if the objectId is outside
#             if objectId not in ins["objectId1"].to_numpy():
#                 # connect the node to the background node, bidirectional
#                 add_edges(edge_indices,background_index,node_index,edge_index)
                
#             # check if there is an object inside current objectId
#             if objectId in ins["objectId2"].to_numpy():
#                 # connect the node to the object inside, bidirectional
#                 current_ins = ins[ins["objectId2"] == objectId]
#                 for childId in current_ins["objectId1"].to_numpy():
#                     child_index = np.where(objectIds == childId)[0][0] + start_object_index
#                     add_edges(edge_indices,node_index,child_index,edge_index)
            
#             node_index += 1
        
#         # add the graph to the data list
#         x = torch.tensor(node_features, dtype=torch.float)
#         edge_index = torch.tensor(edge_indices, dtype=torch.int64)
#         truth_label = torch.tensor(y[i], dtype=torch.int64)

#         # create edge features that all have default value 1
#         edge_features = torch.zeros((edge_index.shape[1],1),dtype=torch.float)

#         data_list += [Data(x=x, edge_index=edge_index, y=truth_label, edge_attr=edge_features)]

#     return data_list



# def create_datalist_klog(kb):
#     datalist = []
#     for i in range(num_of_graphs):
#         #print("Graph: ", i, " out of ", num_of_graphs - 1)
#         # reset params
#         node_index = 0
#         edge_index = 0
        
#         graph_id = kb["bongard"]["problemId"][i]
#         triangles = kb["triangle"][kb["triangle"]["problemId"] == graph_id]
#         squares = kb["square"][kb["square"]["problemId"] == graph_id]
#         circles = kb["circle"][kb["circle"]["problemId"] == graph_id]
#         configs = kb["config"][kb["config"]["problemId"] == graph_id]
#         # triangles up: traingles whith objectId in configs and orient "up"
#         triangles_up = triangles[triangles["objectId"].isin(configs[configs["orient"] == "up"]["objectId"])]
#         # triangles down: traingles whith objectId in configs and orient "down"
#         triangles_down = triangles[triangles["objectId"].isin(configs[configs["orient"] == "down"]["objectId"])]
#         ins = kb["in"][kb["in"]["problemId"] == graph_id]

#         objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
#         objectIds = np.unique(objectIds)

#         # create node feature array
#         num_node_features = 5       # Background/in_node = 0, Square = 1, Circle = 2, Triangle_UP = 3, Triangle_DOWN = 4 
#                                     # 0 if not a node denoting a in relation  1 if a node denoting a in relation
#         num_of_nodes = len(objectIds) + ins.shape[0] + 1 # +1 for the background node
#         node_features = np.zeros((num_of_nodes, num_node_features))

#         # create edge indices array
#         edge_indices = [[],[]]

#         # add the background node
#         background_index = node_index
#         node_features[node_index][:] = np.array([0,0,0,0,0])
#         node_index += 1

#         in_index = len(objectIds) + 1 

#         # set all in relation node features to 1
#         for j in range(ins.shape[0]):
#             node_features[in_index + j][:] = np.array([0,0,0,0,1])
        
#         for obj in objectIds:
#             # add the node to node features
#             if obj in squares["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([1,0,0,0,0])
#             elif obj in circles["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0,1,0,0,0])
#             elif obj in triangles_up["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0,0,1,0,0])
#             elif obj in triangles_down["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0,0,0,1,0])
            
#             # have to connect to background node if the objectId is outside
#             if obj not in ins["objectId1"].to_numpy():
#                 # connect the node to the background node, bidirectional
#                 edge_indices[0].append(background_index)
#                 edge_indices[1].append(node_index)
#                 edge_indices[0].append(node_index)
#                 edge_indices[1].append(background_index)
            
#             # check if there is an object inside current objectId
#             if obj in ins["objectId2"].to_numpy():
#                 # connect the node to the object node inside through an in node
#                 current_ins = ins[ins["objectId2"] == obj]
#                 for childId in current_ins["objectId1"].to_numpy():
#                     child_index = np.where(objectIds == childId)[0][0] + 1

#                     edge_indices[0].append(node_index)
#                     edge_indices[1].append(in_index)
#                     edge_indices[0].append(in_index)
#                     edge_indices[1].append(node_index)

#                     edge_indices[0].append(in_index)
#                     edge_indices[1].append(child_index)
#                     edge_indices[0].append(child_index)
#                     edge_indices[1].append(in_index)

#                     in_index += 1
            
#             node_index += 1
        
#         x = torch.tensor(node_features, dtype=torch.float)
#         edge_index = torch.tensor(edge_indices, dtype=torch.int64)
#         truth_label = torch.tensor(y[i],dtype=torch.int64)
#         edge_features = torch.zeros((edge_index.shape[1],1),dtype=torch.float)
#         data = Data(x=x, edge_index=edge_index, y=truth_label, edge_attr=edge_features)
#         datalist += [data]

#     return datalist
            
# def create_datalist_four(kb):


#     data_list = []
#     for i in range(num_of_graphs):
#         print("Graph: ", i, " out of ", num_of_graphs - 1)
#         # reset params
#         node_index = 0
#         edge_index = 0
#         edge_indices = [[],[]]

#         graph_id = kb["bongard"]["problemId"][i]
#         triangles = kb["triangle"][kb["triangle"]["problemId"] == graph_id]
#         squares = kb["square"][kb["square"]["problemId"] == graph_id]
#         circles = kb["circle"][kb["circle"]["problemId"] == graph_id]
#         configs = kb["config"][kb["config"]["problemId"] == graph_id]
#         # triangles up: traingles whith objectId in configs and orient "up"
#         triangles_up = triangles[triangles["objectId"].isin(configs[configs["orient"] == "up"]["objectId"])]
#         # triangles down: traingles whith objectId in configs and orient "down"
#         triangles_down = triangles[triangles["objectId"].isin(configs[configs["orient"] == "down"]["objectId"])]
#         ins = kb["in"][kb["in"]["problemId"] == graph_id]

#         objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
#         objectIds = np.unique(objectIds) # returns sorted unique elements of an array

#         # init the node features for the objects, for every object there is a node and a there is a global background node
#         num_of_node_features = 1
#         node_features = np.zeros((len(objectIds) + 1, num_of_node_features))

#         # add the background node
#         background_index = node_index
#         node_features[node_index] = 0
#         node_index += 1

#         start_object_index = background_index + 1

#         # a node is connected to the background node if it never appears as objectId1
#         for objectId in objectIds:
#             # add the node to node features
#             if objectId in squares["objectId"].to_numpy():
#                 # square node = 1
#                 node_features[node_index] = 1
#             elif objectId in circles["objectId"].to_numpy():
#                 # circle node = 2
#                 node_features[node_index] = 2
#             elif objectId in triangles_up["objectId"].to_numpy():
#                 # triangle up node = 3
#                 node_features[node_index] = 3
#             elif objectId in triangles_down["objectId"].to_numpy():
#                 # triangle down node = 4
#                 node_features[node_index] = 4

#             # have to connect to background node if the objectId is outside
#             if objectId not in ins["objectId1"].to_numpy():
#                 # connect the node to the background node, bidirectional
#                 edge_indices[0].append(background_index)
#                 edge_indices[1].append(node_index)
#                 edge_index += 1
#                 edge_indices[0].append(node_index)
#                 edge_indices[1].append(background_index)
#                 edge_index += 1
                
#             # check if there is an object inside current objectId
#             if objectId in ins["objectId2"].to_numpy():
#                 # connect the node to the object inside, bidirectional
#                 current_ins = ins[ins["objectId2"] == objectId]
#                 for childId in current_ins["objectId1"].to_numpy():
#                     child_index = np.where(objectIds == childId)[0][0] + start_object_index
#                     edge_indices[0].append(node_index)
#                     edge_indices[1].append(child_index)
#                     edge_index += 1
#                     edge_indices[0].append(child_index)
#                     edge_indices[1].append(node_index)
#                     edge_index += 1
            
#             node_index += 1
        
#         # add the graph to the data list
#         x = torch.tensor(node_features, dtype=torch.int64)
#         edge_index = torch.tensor(edge_indices, dtype=torch.int64)
#         truth_label = torch.tensor(y[i], dtype=torch.int64)

#         data_list += [Data(x=x, edge_index=edge_index, y=truth_label)]

#     return data_list

# def create_datalist_five(kb):
#     datalist = []
#     for i in range(num_of_graphs):
#         #print("Graph: ", i, " out of ", num_of_graphs - 1)
#         # reset params
#         node_index = 0
#         edge_index = 0
        
#         graph_id = kb["bongard"]["problemId"][i]
#         triangles = kb["triangle"][kb["triangle"]["problemId"] == graph_id]
#         squares = kb["square"][kb["square"]["problemId"] == graph_id]
#         circles = kb["circle"][kb["circle"]["problemId"] == graph_id]
#         configs = kb["config"][kb["config"]["problemId"] == graph_id]
#         # triangles up: traingles whith objectId in configs and orient "up"
#         triangles_up = triangles[triangles["objectId"].isin(configs[configs["orient"] == "up"]["objectId"])]
#         # triangles down: traingles whith objectId in configs and orient "down"
#         triangles_down = triangles[triangles["objectId"].isin(configs[configs["orient"] == "down"]["objectId"])]
#         ins = kb["in"][kb["in"]["problemId"] == graph_id]

#         objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
#         objectIds = np.unique(objectIds)

#         # create node feature array
#         num_node_features = 5       
#         num_of_nodes = len(objectIds) + ins.shape[0] + 1 # +1 for the background node
#         node_features = np.zeros((num_of_nodes, num_node_features))

#         # create edge indices array
#         edge_indices = [[],[]]

#         # add the background node
#         background_index = node_index
#         node_features[node_index][:] = np.array([0, 0, 0, 0, 0])
#         node_index += 1

#         in_index = len(objectIds) + 1 

#         # set all in relation node features to 1
#         for j in range(ins.shape[0]):
#             node_features[in_index + j][:] = np.array([0, 0, 0, 0, 1])
        
#         for obj in objectIds:
#             # add the node to node features
#             if obj in squares["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([1, 0, 0, 0, 0])
#             elif obj in circles["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0, 1, 0, 0, 0])
#             elif obj in triangles_up["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0, 0, 1, 0, 0])
#             elif obj in triangles_down["objectId"].to_numpy():
#                 node_features[node_index][:] = np.array([0, 0, 0, 1, 0])
            
#             # have to connect to background node if the objectId is outside
#             if obj not in ins["objectId1"].to_numpy():
#                 # connect the node to the background node, bidirectional
#                 edge_indices[0].append(background_index)
#                 edge_indices[1].append(node_index)
#                 edge_indices[0].append(node_index)
#                 edge_indices[1].append(background_index)
            
#             # check if there is an object inside current objectId
#             if obj in ins["objectId2"].to_numpy():
#                 # connect the node to the object node inside through an in node
#                 current_ins = ins[ins["objectId2"] == obj]
#                 for childId in current_ins["objectId1"].to_numpy():
#                     child_index = np.where(objectIds == childId)[0][0] + 1

#                     edge_indices[0].append(node_index)
#                     edge_indices[1].append(in_index)
#                     edge_indices[0].append(in_index)
#                     edge_indices[1].append(node_index)

#                     edge_indices[0].append(in_index)
#                     edge_indices[1].append(child_index)
#                     edge_indices[0].append(child_index)
#                     edge_indices[1].append(in_index)

#                     in_index += 1
            
#             node_index += 1
        
#         x = torch.tensor(node_features, dtype=torch.int64)
#         edge_index = torch.tensor(edge_indices, dtype=torch.int64)
#         truth_label = torch.tensor(y[i],dtype=torch.int64)
#         data = Data(x=x, edge_index=edge_index, y=truth_label)
#         datalist += [data]

#     return datalist




# ### First representation ###
# data_list = create_datalist(kb)

# # Test First representation
# graph = data_list[0]
# assert torch.equal(torch.tensor(graph.x.shape),torch.tensor([6,4]))
# assert torch.equal(graph.y,torch.tensor(1))
# assert torch.equal(torch.tensor(graph.edge_index.shape),torch.tensor([2,10]))
# assert torch.equal(torch.tensor(graph.edge_attr.shape),torch.tensor([10,1]))

# torch.save(data_list, "datasets/Bongard/HotIn/raw/datalist.pt")

# ### Second representation ###
# data_list = create_datalist(kb)
# data_list = to_node_value_encoding(data_list)

# # Test Second representation
# graph = data_list[0]
# assert torch.equal(torch.tensor(graph.x.shape),torch.tensor([6,1]))
# assert torch.equal(graph.y,torch.tensor(1))
# assert torch.equal(torch.tensor(graph.edge_index.shape),torch.tensor([2,10]))
# assert torch.equal(torch.tensor(graph.edge_attr.shape),torch.tensor([10,1]))

# torch.save(data_list, "datasets/Bongard/ValIn/raw/datalist.pt")

# ### Third representation ###
# data_list = create_datalist_edges(kb)

# # Test Third representation
# graph = data_list[0]
# assert torch.equal(torch.tensor(graph.x.shape),torch.tensor([5,4]))
# assert torch.equal(graph.y,torch.tensor(1))
# assert torch.equal(torch.tensor(graph.edge_index.shape),torch.tensor([2,20]))
# assert torch.equal(torch.tensor(graph.edge_attr.shape),torch.tensor([20,1]))

# torch.save(data_list, "datasets/Bongard/HotEdgesFull/raw/datalist.pt")

# ### Fourth representation ###
# data_list = create_datalist_edges(kb)
# data_list = to_node_value_encoding(data_list)

# # Test Fourth representation
# graph = data_list[0]
# assert torch.equal(torch.tensor(graph.x.shape),torch.tensor([5,1]))
# assert torch.equal(graph.y,torch.tensor(1))
# assert torch.equal(torch.tensor(graph.edge_index.shape),torch.tensor([2,20]))
# assert torch.equal(torch.tensor(graph.edge_attr.shape),torch.tensor([20,1]))

# torch.save(data_list, "datasets/Bongard/ValEdgesFull/raw/datalist.pt")


# ### Fifth representation ###
# data_list = create_datalist_klog(kb)

# # Test Fifth representation
# graph = data_list[0]
# assert torch.equal(torch.tensor(graph.x.shape),torch.tensor([8,5]))
# assert torch.equal(graph.y,torch.tensor(1))
# assert torch.equal(torch.tensor(graph.edge_index.shape),torch.tensor([2,14]))
# assert torch.equal(torch.tensor(graph.edge_attr.shape),torch.tensor([14,1]))

# torch.save(data_list, "datasets/Bongard/HotKlog/raw/datalist.pt")


# ### Sixth representation ###
# data_list = create_datalist_klog(kb)
# data_list = to_node_value_encoding(data_list)

# # Test Sixth representation
# graph = data_list[0]
# assert torch.equal(torch.tensor(graph.x.shape),torch.tensor([8,1]))
# assert torch.equal(graph.y,torch.tensor(1))
# assert torch.equal(torch.tensor(graph.edge_index.shape),torch.tensor([2,14]))
# assert torch.equal(torch.tensor(graph.edge_attr.shape),torch.tensor([14,1]))

# torch.save(data_list, "datasets/Bongard/ValKlog/raw/datalist.pt")


# ### Seventh representation ###
# # No frame node 

# data_list = create_datalist(kb)
# for i in range(len(data_list)):
#     graph = data_list[i]
#     x = graph.x
#     edge_indices = graph.edge_index
#     new_x = x[1:]
#     new_edge_indices = [[],[]]
#     for j in range(len(edge_indices[0])):
#         if edge_indices[0][j] != 0 and edge_indices[1][j] != 0:
#             new_edge_indices[0].append(edge_indices[0][j]-1)
#             new_edge_indices[1].append(edge_indices[1][j]-1)
#     graph.x = new_x
#     graph.edge_index = torch.tensor(new_edge_indices, dtype=torch.long)
#     # graph.edge_index[0] -= 1
#     # graph.edge_index[1] -= 1
#     if i == 0:
#         print(graph.x)
#         print(graph.edge_index)
#         print(graph.edge_attr)

# torch.save(data_list, "datasets/Bongard/NoFrame/raw/datalist.pt")


# ### eight representation ###
# # fixed nodes only representation

# # datalist is still the same as the NoFrame representation
# # add a frame node that is connected to all other nodes

# for i in range(len(data_list)):
#     graph = data_list[i]
#     x = graph.x
#     edge_indices = graph.edge_index
#     # create new x
#     new_x = torch.zeros((x.shape[0]+1, x.shape[1]), dtype=torch.float)
#     new_x[1:] = x
#     new_x[0] = torch.tensor([0,0,0,0])
#     # create new edge indices
#     new_edge_indices = [[],[]]
#     for j in range(len(edge_indices[0])):
#         new_edge_indices[0].append(edge_indices[0][j]+1)
#         new_edge_indices[1].append(edge_indices[1][j]+1)
#     for j in range(1, new_x.shape[0]):
#         new_edge_indices[0].append(0)
#         new_edge_indices[1].append(j)
#         new_edge_indices[0].append(j)
#         new_edge_indices[1].append(0)
#     graph.x = new_x
#     graph.edge_index = torch.tensor(new_edge_indices, dtype=torch.long)

# torch.save(data_list, "datasets/Bongard/NodesOnly/raw/datalist.pt")

    
### Ninth representation ###
# correct KLog representation
# in between frame node and shape nodes add a node called part of node
# inbetween 2 shape nodes add a node called in node

    
            



            
    

    

    

# ### Seventh representation ###
# # Heterogeneous graph
    
# data_list = []
# for i in range(num_of_graphs):
#     # reset params
#     node_index = 0
#     edge_index = 0
#     edge_indices = [[],[]]

#     graph_id = kb["bongard"]["problemId"][i]
#     triangles = kb["triangle"][kb["triangle"]["problemId"] == graph_id]
#     squares = kb["square"][kb["square"]["problemId"] == graph_id]
#     circles = kb["circle"][kb["circle"]["problemId"] == graph_id]
#     configs = kb["config"][kb["config"]["problemId"] == graph_id]
#     # triangles up: traingles whith objectId in configs and orient "up"
#     triangles_up = triangles[triangles["objectId"].isin(configs[configs["orient"] == "up"]["objectId"])]
#     # triangles down: traingles whith objectId in configs and orient "down"
#     triangles_down = triangles[triangles["objectId"].isin(configs[configs["orient"] == "down"]["objectId"])]
#     ins = kb["in"][kb["in"]["problemId"] == graph_id]

#     objectIds = np.concatenate((triangles["objectId"].to_numpy(), squares["objectId"].to_numpy(), circles["objectId"].to_numpy()))
#     objectIds = np.unique(objectIds) # returns sorted unique elements of an array


#     # Create the heterogeneous graph
#     data = HeteroData()
#     obj_mapping = {obj:i for i,obj in enumerate(objectIds)}

#     # define the frame features
#     amount_of_neighbors = len(objectIds)
#     neighbors = set(objectIds)
#     for objectId in objectIds:
#         if objectId in ins["objectId1"].to_numpy():
#             amount_of_neighbors -= 1
#             neighbors.discard(objectId)

#     data['frame'].x = torch.tensor([[amount_of_neighbors]], dtype=torch.float)
    
#     # define the shape features 
#     shape_node_features = np.zeros((len(objectIds), 4))
#     for i,objectId in enumerate(objectIds):
#         if objectId in squares["objectId"].to_numpy():
#             shape_node_features[i][:] = np.array([1, 0, 0, 0])
#         elif objectId in circles["objectId"].to_numpy():
#             shape_node_features[i][:] = np.array([0, 1, 0, 0])
#         elif objectId in triangles_up["objectId"].to_numpy():
#             shape_node_features[i][:] = np.array([0, 0, 1, 0])
#         elif objectId in triangles_down["objectId"].to_numpy():
#             shape_node_features[i][:] = np.array([0, 0, 0, 1])
    
#     data['shape'].x = torch.tensor(shape_node_features, dtype=torch.float)

#     # define the connected edges and features
#     connected_edges = [[],[]]
#     for Id in neighbors:
#         connected_edges[0].append(0)
#         connected_edges[1].append(obj_mapping[Id])

#     data["frame","connected","shape"].edge_index = torch.tensor(connected_edges, dtype=torch.long)
#     data["frame","connected","shape"].edge_attr = torch.zeros((len(connected_edges[0]),1), dtype=torch.float)

#     # define the in edges and features
#     in_edges = [[],[]]
#     for objectId in objectIds:
#         if objectId in ins["objectId1"].to_numpy():
#             current_ins = ins[ins["objectId1"] == objectId]
#             for parentId in current_ins["objectId2"].to_numpy():
#                 parent_index = obj_mapping[parentId]
#                 in_edges[0].append(parent_index)
#                 in_edges[1].append(obj_mapping[objectId])
#     data["shape","in","shape"].edge_index = torch.tensor(in_edges, dtype=torch.long)
#     data["shape","in","shape"].edge_attr = torch.zeros((len(in_edges[0]),1), dtype=torch.float)

#     # define the ground truth label
#     data.y = torch.tensor(y[i], dtype=torch.long)
    
#     data_list += [data]

# # Test Seventh representation
# graph = data_list[0]


# for i in range(len(data_list)):
#     assert "frame" in data_list[i].x_dict.keys()

# torch.save(data_list, "datasets/Bongard/Heterogeneous/raw/datalist.pt")

