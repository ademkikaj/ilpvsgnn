import random
import numpy as np
import pandas as pd

MIN_GRAPH_SIZE = 2
MAX_GRAPH_SIZE = 5
COLOURS = ['green', 'red']
THRESHOLD = 0.5

def gen_graph(min_node=MIN_GRAPH_SIZE, max_node=MAX_GRAPH_SIZE, prob=THRESHOLD):
    n = random.randint(min_node, max_node)
    colours = [random_colour() for _ in range(n)]
    edges = make_edges(n, prob=prob)
    return colours, edges

def is_connected(edges, i, j):
    visited, neighbour = set(), [i]
    while neighbour:
        neighbour = [j for n in neighbour for j in range(len(edges)) if edges[n][j] and j not in visited]
        visited.update(neighbour)
        if j in neighbour:
            return True
    return False

def path(edges, s, d):
    visited = [False] * len(edges)
    queue = []
    queue.append([s, [s]])

    while queue:
        [n, path] = queue.pop(0)
        for i in range(len(edges)):
            if edges[n][i] and not visited[i]:
                queue.append([i, path + [i]])
                visited[i] = True
                if i == d:
                    return path + [i]
    return False

def random_colour():
    return random.choice(COLOURS)


def make_edges(n, prob=THRESHOLD):
    edges = np.random.rand(n, n)
    edges[edges > prob] = 1
    edges[edges <= prob] = 0
    for i in range(n):
        edges[i][i] = 0
    return edges


def gen_pos(k):
    while True:
        colours, edges = gen_graph(min_node=2, prob=0.9)
        [start] = random.sample(list(range(len(edges))), 1)
        n = random.randint(1, len(edges))
        i, j, unvisited = start, start, set([i for i in range(len(edges))])
        while n:
            [j] = random.sample(unvisited, 1)
            unvisited.remove(j)
            edges[i][j] = 1
            i = j
            n -= 1
        edges[j][start] = 1
        return [k, colours, edges], f"f(n_{k}_{start})"


def gen_neg(k):
    colours, edges = gen_graph(min_node=2, prob=0.9)
    [i] = random.sample(list(range(len(edges))), 1)
    while path(edges, i, i):
        cyclic_path = path(edges, i, i)
        u = random.randint(0, len(cyclic_path) - 1)
        edges[cyclic_path[u], cyclic_path[(u + 1) % len(cyclic_path)]] = 0
    assert not is_connected(edges, i, i)
    return [k, colours, edges], f"f(n_{k}_{i})"


ouput_path = "Benchmark/cyclic/Cyclic"
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
    
for i in range(100,200):
    neg = gen_neg(i)
    id = neg[0]
    for j in range(len(id[1])):
        nodes = pd.concat([nodes, pd.DataFrame(columns=["id","node_id", "color"], data=[{"id": id[0], "node_id": j, "color": id[1][j]}])], ignore_index=True)
        for k in range(len(id[2])):
            if id[2][j][k] == 1:
                edges = pd.concat([edges, pd.DataFrame(columns=["id","node_1", "node_2"], data=[{"id": id[0], "node_1": j, "node_2": k}])], ignore_index=True)
    cyclic = pd.concat([cyclic, pd.DataFrame(columns=["id","class"], data=[{"id": id[0], "class": "neg"}])], ignore_index=True)

nodes.to_csv(f"{ouput_path}/nodes.csv", index=False)
edges.to_csv(f"{ouput_path}/edges.csv", index=False)
cyclic.to_csv(f"{ouput_path}/cyclic.csv", index=False)
