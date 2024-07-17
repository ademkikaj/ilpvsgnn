from GraphConversion import GraphConversion
from torch_geometric.utils import to_dense_adj

class toKernel(GraphConversion):

    def node_edge(self,output_path,graphs):

        data_list = {
            "Adjacency Matrix": [],
            "Node Labels": [],
            "Graph Labels": []
        }

        for i in range(self.num_of_graphs):
            graph = graphs[i]
            graph_id = self.kb[self.dataset_name][self.problem_key].unique()[i]
            current_objects = self.get_current_objects(graph_id)

            
            # turn a graph in adjacency matrix
            adjacency = to_dense_adj(graph.edge_index).squeeze(0).int().numpy()

            # node labels
            node_labels = ["white_king","white_rook","black_king"]
            node_labels = {i:node_labels[i] for i in range(len(node_labels))}

            # add results
            data_list["Adjacency Matrix"].append(adjacency)
            data_list["Node Labels"].append(node_labels)
            data_list["Graph Labels"].append(self.y[i])
        
        

        return data_list


            


    
