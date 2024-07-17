import random
import numpy as np
import pandas as pd

MAX_GRAPH_SIZE = 3
COLOURS = ['green', 'red']
THRESHOLD = 0.5


def random_colour():
    return random.choice(COLOURS)

def make_edges(n, prob=THRESHOLD):
    edges = np.random.rand(n, n)
    edges[edges > prob] = 1
    edges[edges <= prob] = 0
    for i in range(n):
        edges[i][i] = 0
    return edges


def gen_graph(min_node=1, max_node=MAX_GRAPH_SIZE, prob=THRESHOLD):
    n = random.randint(min_node, max_node)
    colours = [random_colour() for _ in range(n)]
    edges = make_edges(n, prob=prob)
    return colours, edges


def gen_pos(k):
    colours, edges = gen_graph(min_node=2)
    [i, j] = random.sample(list(range(len(edges))), 2)
    edges[i][j] = 1
    colours[j] = colours[i]
    return [k, colours, edges], f"f(n_{k}_{i})"


def gen_neg(k):
    while True:
        colours, edges = gen_graph(min_node=2)
        [i] = random.sample(list(range(len(edges))), 1)
        bad_neighbors = [j for j in range(len(edges)) if edges[i][j] and colours[j] == colours[i]]
        if not bad_neighbors:
            return [k, colours, edges], f"f(n_{k}_{i})"
        

output_path = "Benchmark/color/relational"
nodes = pd.DataFrame(columns=["id","node_id", "color"], data=[])
edges = pd.DataFrame(columns=["id","node_1", "node_2"], data=[])
color = pd.DataFrame(columns=["id","class"], data=[])


for i in range(100):
    pos = gen_pos(i)
    id = pos[0]
    for j in range(len(id[1])):
        nodes = pd.concat([nodes, pd.DataFrame(columns=["id","node_id", "color"], data=[{"id": id[0], "node_id": j, "color": id[1][j]}])], ignore_index=True)
        for k in range(len(id[2])):
            if id[2][j][k] == 1:
                edges = pd.concat([edges, pd.DataFrame(columns=["id","node_1", "node_2"], data=[{"id": id[0], "node_1": j, "node_2": k}])], ignore_index=True)
    color = pd.concat([color, pd.DataFrame(columns=["id","class"], data=[{"id": id[0], "class": "pos"}])], ignore_index=True)
    
for i in range(100,200):
    neg = gen_neg(i)
    id = neg[0]
    node = int(neg[1][-2])
    for j in range(len(id[1])):
        nodes = pd.concat([nodes, pd.DataFrame(columns=["id","node_id", "color"], data=[{"id": id[0], "node_id": j, "color": id[1][j]}])], ignore_index=True)
        for k in range(len(id[2])):
            if id[2][j][k] == 1:
                edges = pd.concat([edges, pd.DataFrame(columns=["id","node_1", "node_2"], data=[{"id": id[0], "node_1": j, "node_2": k}])], ignore_index=True)
    color = pd.concat([color, pd.DataFrame(columns=["id","class"], data=[{"id": id[0], "class": "neg"}])], ignore_index=True)

nodes.to_csv(f"{output_path}/nodes.csv", index=False)
edges.to_csv(f"{output_path}/edges.csv", index=False)
color.to_csv(f"{output_path}/color.csv", index=False)
