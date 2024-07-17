import networkx as nx
import random
from torch_geometric.data import Data
import torch
import pandas as pd
import os


def generate_graph(min_size, max_size, is_positive):
    # Generates a single graph with specified properties.
    
    # Args:
    # min_size (int): Minimum number of nodes in the graph.
    # max_size (int): Maximum number of nodes in the graph.
    # is_positive (bool): Flag indicating whether the graph should have a 4-node cycle (positive) or not (negative).
    
    # Returns:
    # G (networkx.Graph): The generated graph.
    
    # Randomly choose the number of nodes for the graph
    num_nodes = random.randint(min_size, max_size)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Assign random colors to nodes
    colors = ["red", "green", "blue", "yellow"]
    color_map = {node: random.choice(colors) for node in G.nodes()}
    nx.set_node_attributes(G, color_map, 'color')
    
    # Generating the graph structure
    if is_positive:
        # Create a cycle of 4 nodes if positive
        cycle_nodes = random.sample(G.nodes(), 4)
        nx.add_cycle(G, cycle_nodes)
        # Adding additional random edges to ensure connectivity and complexity
        for _ in range(int(1.5 * num_nodes)):  # Number of additional edges
            n1, n2 = random.sample(G.nodes(), 2)
            G.add_edge(n1, n2)
    else:
        # Add edges but avoid creating a cycle of exactly 4 nodes, when clearing also add the colors attribute
        while True:
            G.clear()
            G.add_nodes_from(range(num_nodes))
            color_map = {node: random.choice(colors) for node in G.nodes()}
            nx.set_node_attributes(G, color_map, 'color')
            edges = [(random.choice(list(G.nodes())), random.choice(list(G.nodes()))) for _ in range(int(1.5 * num_nodes))]
            G.add_edges_from(edges)
            if not any(len(cycle) == 4 for cycle in nx.cycle_basis(G)):
                break

    return G

def generate_graph_dataset(num_graphs, positive_ratio, min_size, max_size):
    # Generates a dataset of graphs with labels indicating whether they contain a specific substructure (4-node cycle).
    
    # Args:
    # num_graphs (int): Number of graphs to generate.
    # positive_ratio (float): Proportion of graphs that should be positive (have a 4-node cycle).
    # min_size (int): Minimum number of nodes in each graph.
    # max_size (int): Maximum number of nodes in each graph.
    
    # Returns:
    # list: A list of tuples where each tuple contains a networkx Graph and a label (0 for negative, 1 for positive).
    
    dataset = []
    num_positive = int(num_graphs * positive_ratio)
    num_negative = num_graphs - num_positive
    
    # Generate positive graphs with a 4-node cycle
    for _ in range(num_positive):
        graph = generate_graph(min_size, max_size, is_positive=True)
        dataset.append((graph, 1))
    
    # Generate negative graphs without a 4-node cycle
    for _ in range(num_negative):
        graph = generate_graph(min_size, max_size, is_positive=False)
        dataset.append((graph, 0))
    
    # Shuffle the dataset to mix positive and negative graphs
    random.shuffle(dataset)
    return dataset

def convert_to_pyg_data(graph, label):
    # Node features: assuming node attribute 'color' is categorical and mapped to integers
    color_mapping = {'red': 0, 'green': 1, 'blue': 2, 'yellow': 3}
    node_features = torch.tensor([color_mapping[data['color']] for _, data in graph.nodes(data=True)], dtype=torch.long)
    
    # Convert node features to one-hot encoding if necessary
    node_features = torch.nn.functional.one_hot(node_features, num_classes=len(color_mapping))
    
    # Edges
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    
    # Create PyG Data object
    data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    
    return data

def generate_tables(dataset):
    graph_data = []
    node_data = []
    edge_data = []
    
    for graph_id, (graph, label) in enumerate(dataset):
        # Populate graph table
        graph_data.append({'id': graph_id, 'class': label})
        
        # Populate node table
        for node in graph.nodes(data=True):
            node_id = node[0]
            color = node[1]['color']
            node_data.append({'id': graph_id, 'node_id': node_id, 'color': color})
        
        # Populate edge table
        for edge in graph.edges():
            node_id_1, node_id_2 = edge
            edge_data.append({'id': graph_id, 'node_1': node_id_1, 'node_2': node_id_2})
    
    # Convert lists to DataFrames
    graph_df = pd.DataFrame(graph_data)
    node_df = pd.DataFrame(node_data)
    edge_df = pd.DataFrame(edge_data)
    
    return graph_df, node_df, edge_df



if __name__ == "__main__":
    num_graphs = 100
    positive_ratio = 0.5
    min_size = 4
    max_size = 4
    
    dataset = generate_graph_dataset(num_graphs, positive_ratio, min_size, max_size)
    for i, (graph, label) in enumerate(dataset):
        print(f"Graph {i+1}: {'Positive' if label == 1 else 'Negative'}")
        print("Nodes:", graph.nodes())
        print("Edges:", graph.edges())
        print()
    
    graph_df, node_df, edge_df = generate_tables(dataset)

    base_path = os.path.join("docker", "Benchmark","cyclic","relational")
    graph_df.to_csv(os.path.join(base_path, "cyclic.csv"), index=False)
    node_df.to_csv(os.path.join(base_path, "node.csv"), index=False)
    edge_df.to_csv(os.path.join(base_path, "edge.csv"), index=False)

    dataset = [convert_to_pyg_data(graph, label) for graph, label in dataset]

    torch.save(dataset, 'graph_dataset.pt')
    
    