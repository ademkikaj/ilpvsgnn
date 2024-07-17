import torch
from torch_geometric.data import Data
from ..GraphConversion import GraphConversion
import numpy as np

class toGraph(GraphConversion):

    def node_only(self):
        data_list = []
        for graph_index  in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key].unique()[graph_index]
            spl = graph_id.split("_")
            prefix = graph_id[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]

            num_node_features = len(self.kb["nodes"]["color"].unique()) + 1
            #num_node_features = 
            num_nodes = len(current_objects["nodes"]) + len(current_objects["edges"])

            node_features = np.zeros((num_nodes, num_node_features))

            node_index = 0
            for _,row in current_objects["nodes"].iterrows():
                if row["color"] == "red":
                    node_features[node_index,0] = 1
                elif row["color"] == "green":
                    node_features[node_index,1] = 1
                elif row["color"] == "blue":
                    node_features[node_index,2] = 1
                elif row["color"] == "yellow":
                    node_features[node_index,3] = 1
                node_index += 1
            
            for _,row in current_objects["edges"].iterrows():
                node_features[node_index,-1] = 1
                node_index += 1
            
            # fully connect the nodes
            edge_indices = [[],[]]
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        self.add_edges(edge_indices, i, j)

            new_graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.ones((len(edge_indices[0]),1), dtype=torch.float),
                    y=torch.tensor([self.y[graph_index]], dtype=torch.int64)
                )
            
            #assert new_graph.is_directed() == True
            assert new_graph.has_self_loops() == False
            data_list.append(new_graph)
            
        return data_list

    def node_edge(self):
        data_list = []

        for graph_index  in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key].unique()[graph_index]
            spl = graph_id.split("_")
            prefix = graph_id[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]


            num_node_features = len(self.kb["nodes"]["color"].unique())
            # num_node_features = 4
            num_nodes = len(current_objects["nodes"])

            # assert num_node_features == num_nodes

            node_features = np.zeros((num_nodes, num_node_features))

            node_index = 0
            for _,row in current_objects["nodes"].iterrows():
                if row["color"] == "red":
                    node_features[node_index,0] = 1
                elif row["color"] == "green":
                    node_features[node_index,1] = 1
                elif row["color"] == "blue":
                    node_features[node_index,2] = 1
                elif row["color"] == "yellow":
                    node_features[node_index,3] = 1
                node_index += 1
            
            edge_indices = [[],[]]

            for _,row in current_objects["edges"].iterrows():
                edge_index_1 = int(row["node_1"].split("_")[-1])
                edge_index_2 = int(row["node_2"].split("_")[-1])
                edge_indices[0].append(edge_index_1)
                edge_indices[1].append(edge_index_2)
            
            new_graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.ones((len(edge_indices[0]),1), dtype=torch.float),
                    y=torch.tensor([self.y[graph_index]], dtype=torch.int64)
                )
            #assert new_graph.is_directed() == True
            assert new_graph.has_self_loops() == False
            data_list.append(new_graph)
        return data_list

    def edge_based(self):
        data_list = []

        for graph_index  in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key].unique()[graph_index]
            spl = graph_id.split("_")
            prefix = graph_id[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]


            num_node_features = len(self.kb["nodes"]["color"].unique()) + 1 # 1 is for instance node
            num_nodes = len(current_objects["nodes"]) + len(self.kb["nodes"]["color"].unique())

            node_features = np.zeros((num_nodes, num_node_features))

            # create the color nodes
            color_nodes = {}
            node_index = 0
            # for i,color in enumerate(self.kb["nodes"]["color"].unique()):
            #     node_features[node_index,i] = 1
            #     color_nodes[color] = node_index
            #     node_index += 1
            node_features[node_index,0] = 1
            color_nodes["red"] = node_index
            node_index += 1
            node_features[node_index,1] = 1
            color_nodes["green"] = node_index
            node_index += 1
            
            edge_indices = [[],[]]
            for _,row in current_objects["nodes"].iterrows():
                node_features[node_index,-1] = 1
                self.add_edges(edge_indices, node_index, color_nodes[row["color"]])
                node_index += 1
    
            for _,row in current_objects["edges"].iterrows():
                edge_index_1 = int(row["node_1"].split("_")[-1])
                edge_index_2 = int(row["node_2"].split("_")[-1])
                edge_indices[0].append(edge_index_1)
                edge_indices[1].append(edge_index_2)
                            
            new_graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.ones((len(edge_indices[0]),1), dtype=torch.float),
                    y=torch.tensor([self.y[graph_index]], dtype=torch.int64)
                )

            #assert new_graph.is_directed() == True
            assert new_graph.has_self_loops() == False
            data_list.append(new_graph)
        return data_list

    def Klog(self):
        data_list = []

        for graph_index  in range(self.num_of_graphs):

            graph_id = self.kb[self.dataset_name][self.problem_key].unique()[graph_index]
            spl = graph_id.split("_")
            prefix = graph_id[:-2]
            prefix = f"n_{spl[1]}_"
            current_objects = {}
            current_objects["nodes"] = self.kb["nodes"][self.kb["nodes"]["node_id"].str.startswith(prefix)]
            current_objects["edges"] = self.kb["edges"][self.kb["edges"]["node_1"].str.startswith(prefix) | self.kb["edges"]["node_2"].str.startswith(prefix)]

            num_node_features = len(self.kb["nodes"]["color"].unique()) + 1
            num_nodes = len(current_objects["nodes"]) + len(current_objects["edges"])
            node_features = np.zeros((num_nodes, num_node_features))

            node_index = 0
            for _,row in current_objects["nodes"].iterrows():
                if row["color"] == "red":
                    node_features[node_index,0] = 1
                elif row["color"] == "green":
                    node_features[node_index,1] = 1
                elif row["color"] == "blue":
                    node_features[node_index,2] = 1
                elif row["color"] == "yellow":
                    node_features[node_index,3] = 1
                node_index += 1
            
            edge_indices = [[],[]]
            for _,row in current_objects["edges"].iterrows():
                node_features[node_index,-1] = 1

                edge_index_1 = int(row["node_1"].split("_")[-1])
                edge_index_2 = int(row["node_2"].split("_")[-1])

                edge_indices[0].append(edge_index_1)
                edge_indices[1].append(node_index)

                edge_indices[0].append(node_index)
                edge_indices[1].append(edge_index_2)

                node_index += 1
            
            new_graph = Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.ones((len(edge_indices[0]),1), dtype=torch.float),
                    y=torch.tensor([self.y[graph_index]], dtype=torch.int64)
                )
            #assert new_graph.is_directed() == True
            assert new_graph.has_self_loops() == False
            data_list.append(new_graph)
            

        return data_list