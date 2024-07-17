import torch
from torch_geometric.data import Data
from ..GraphConversion import GraphConversion
import numpy as np

class toGraph(GraphConversion):

    def node_edge(self):

        data_list = []
        for graph_index in range(self.num_of_graphs):
            graph_id = self.kb[self.dataset_name][self.problem_key].unique()[graph_index]
            current_objects = self.get_current_objects(graph_id)

            num_node_features = len(self.kb["nodes"]["color"].unique()) 
            num_nodes = len(current_objects["nodes"])

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
                edge_indices[0].append(row["node_1"])
                edge_indices[1].append(row["node_2"])
            
            data_list.append(
                Data(
                    x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(edge_indices, dtype=torch.long),
                    edge_attr=torch.ones((len(edge_indices[0]),1), dtype=torch.float),
                    y=torch.tensor([self.y[graph_index]], dtype=torch.int64)
                )
            )
        return data_list