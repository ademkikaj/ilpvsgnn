import random
import numpy as np
import pandas as pd

# Constants for graph size and probability threshold
MIN_GRAPH_SIZE = 10
MAX_GRAPH_SIZE = 20
THRESHOLD = 0.9
COLOURS = ['Red', 'Green', 'Blue', 'Yellow']  # Example colour choices
COLOURS = ['Red','Green','Blue','Yellow']
COLOURS = ['RED']

ouput_path = "Benchmark/cyclic/cyclic1"

def gen_graph(min_node=MIN_GRAPH_SIZE, max_node=MAX_GRAPH_SIZE, prob=THRESHOLD):
    n = random.randint(min_node, max_node)
    colours = [random_colour() for _ in range(n)]
    edges = make_edges(n, prob)
    return colours, edges

def random_colour():
    return random.choice(COLOURS)

def make_edges(n, prob=THRESHOLD):
    edges = np.random.rand(n, n) < prob
    np.fill_diagonal(edges, 0)
    return edges

def has_cycle_of_length(edges, length):
    def dfs(current, start, count, visited):
        if count == length and current == start:
            return True
        visited.add(current)
        for neighbor in range(len(edges)):
            if edges[current][neighbor] and (neighbor not in visited or (neighbor == start and count == length - 1)):
                if dfs(neighbor, start, count + 1, visited):
                    return True
        visited.remove(current)
        return False

    for node in range(len(edges)):
        if dfs(node, node, 0, set()):
            return True
    return False

def break_cycle(edges, length):
    for start in range(len(edges)):
        if break_cycle_from_node(edges, start, start, 0, length, set()):
            return

def break_cycle_from_node(edges, current, start, count, length, visited):
    if count == length and current == start:
        return True
    visited.add(current)
    for neighbor in range(len(edges)):
        if edges[current][neighbor]:
            if neighbor not in visited or (neighbor == start and count == length - 1):
                if break_cycle_from_node(edges, neighbor, start, count + 1, length, visited):
                    edges[current][neighbor] = 0  # Break the cycle
                    return True
    visited.remove(current)
    return False

def gen_pos(k):
    print(k)
    while True:
        colours, edges = gen_graph(min_node=MIN_GRAPH_SIZE, max_node=MAX_GRAPH_SIZE, prob=0.9)  # Adjust number of nodes to exactly 4 for simplicity
        nodes = list(range(4))
        random.shuffle(nodes)
        for i in range(4):
            edges[nodes[i]][nodes[(i + 1) % 4]] = 1  # Create a 4-node cycle
        
        if has_cycle_of_length(edges, 4):  # Check for 4-node cycle
            return [k, colours, edges], f"f(n_{k}_{nodes[0]})"
        print(k + 1)

def gen_neg(k):
    while True:
        colours, edges = gen_graph(min_node=MIN_GRAPH_SIZE, max_node=MAX_GRAPH_SIZE, prob=0.9)  # Ensure enough nodes for complexity
        if has_cycle_of_length(edges, 4):
            break_cycle(edges, 4)
        
        if not has_cycle_of_length(edges, 4):  # Verify no 4-node cycle exists
            return [k, colours, edges], f"f(n_{k})"


nodes = pd.DataFrame(columns=["id","node_id", "color"], data=[])
edges = pd.DataFrame(columns=["id","node_1", "node_2"], data=[])
cyclic = pd.DataFrame(columns=["id","class"], data=[])

for i in range(100):
    pos = gen_pos(i)
    id = pos[0]
    for j in range(len(id[1])):
        nodes = pd.concat([nodes, pd.DataFrame(columns=["id","node_id", "color"], data=[{"id": id[0], "node_id": j, "color": id[1][j]}])], ignore_index=True)
        for k in range(len(id[2])):
            if id[2][j][k] == 1:
                edges = pd.concat([edges, pd.DataFrame(columns=["id","node_1", "node_2"], data=[{"id": id[0], "node_1": j, "node_2": k}])], ignore_index=True)
    cyclic = pd.concat([cyclic, pd.DataFrame(columns=["id","class"], data=[{"id": id[0], "class": "pos"}])], ignore_index=True)
print("Positives generated")
for i in range(100,200):
    neg = gen_neg(i)
    id = neg[0]
    for j in range(len(id[1])):
        nodes = pd.concat([nodes, pd.DataFrame(columns=["id","node_id", "color"], data=[{"id": id[0], "node_id": j, "color": id[1][j]}])], ignore_index=True)
        for k in range(len(id[2])):
            if id[2][j][k] == 1:
                edges = pd.concat([edges, pd.DataFrame(columns=["id","node_1", "node_2"], data=[{"id": id[0], "node_1": j, "node_2": k}])], ignore_index=True)
    cyclic = pd.concat([cyclic, pd.DataFrame(columns=["id","class"], data=[{"id": id[0], "class": "neg"}])], ignore_index=True)
print("Negatives generated")

nodes.to_csv(f"{ouput_path}/nodes.csv", index=False)
edges.to_csv(f"{ouput_path}/edges.csv", index=False)
cyclic.to_csv(f"{ouput_path}/cyclic.csv", index=False)