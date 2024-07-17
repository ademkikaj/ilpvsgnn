import random
import numpy as np
import pandas as pd

MIN_GRAPH_SIZE = 4
MAX_GRAPH_SIZE = 4
COLOURS = ['red','green']
THRESHOLD = 0.5

def gen_graph(min_node=MIN_GRAPH_SIZE, max_node=MAX_GRAPH_SIZE, prob=THRESHOLD,n_color=2):
    n = random.randint(min_node, max_node)
    colours = [random_colour(n_color) for _ in range(n)]
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

def random_colour(n_color):
    return random.choice(COLOURS[:n_color])


def make_edges(n, prob=THRESHOLD):
    edges = np.random.rand(n, n)
    edges[edges > prob] = 1
    edges[edges <= prob] = 0
    for i in range(n):
        edges[i][i] = 0
    return edges


def gen_pos(k,graph_nodes,cycle_size,n_color):
    while True:
        colours, edges = gen_graph(min_node=graph_nodes,max_node=graph_nodes, prob=0.9,n_color=n_color)
        [start] = random.sample(list(range(len(edges))), 1)
        n = random.randint(1, len(edges))
        n = cycle_size
        i, j, unvisited = start, start, set([i for i in range(len(edges))])
        unvisited.remove(i)
        while n:
            [j] = random.sample(unvisited, 1)
            unvisited.remove(j)
            edges[i][j] = 1
            i = j
            n -= 1
        edges[j][start] = 1
        return [k, colours, edges], f"n_{k}_{start}"


def gen_neg(k,graph_nodes,n_color):
    colours, edges = gen_graph(min_node=graph_nodes,max_node=graph_nodes, prob=0.9,n_color=n_color)
    [i] = random.sample(list(range(len(edges))), 1)
    while path(edges, i, i):
        cyclic_path = path(edges, i, i)
        u = random.randint(0, len(cyclic_path) - 1)
        edges[cyclic_path[u], cyclic_path[(u + 1) % len(cyclic_path)]] = 0
    assert not is_connected(edges, i, i)
    return [k, colours, edges], f"n_{k}_{i}"



def generate_relational_data(output_path, num_graphs=200, min_graph=4, max_graph=4,cycle_size=4,dataset_name=None,n_color=2, THRESHOLD=0.5):

    MIN_GRAPH_SIZE = min_graph
    MAX_GRAPH_SIZE = max_graph

    #ouput_path = "docker/Benchmark/cyclic/relational"

    nodes = pd.DataFrame(columns=["node_id", "color"], data=[])
    edges = pd.DataFrame(columns=["node_1", "node_2"], data=[])
    cyclic = pd.DataFrame(columns=["id","class"], data=[])

    for index in range(num_graphs//2):
        pos = gen_pos(index,min_graph,cycle_size=cycle_size,n_color=n_color)
        [k,colours,edge] = pos[0]
        target_node = pos[1]
        for i,c in enumerate(colours):
            node = f"n_{k}_{i}"
            nodes = pd.concat([nodes, pd.DataFrame(columns=["node_id", "color"], data=[{"node_id": node, "color": c}])], ignore_index=True)
        for i in range(len(edge)):
            for j in range(len(edge)):
                if edge[i][j] == 1:
                    edges = pd.concat([edges, pd.DataFrame(columns=["node_1", "node_2"], data=[{"node_1": f"n_{k}_{i}", "node_2": f"n_{k}_{j}"}])], ignore_index=True)
        cyclic = pd.concat([cyclic, pd.DataFrame(columns=["id","class"], data=[{"id": target_node, "class": "pos"}])], ignore_index=True)
        
    for index in range(num_graphs//2,num_graphs):
        neg = gen_neg(index,min_graph,n_color=n_color)
        [k,colours,edge] = neg[0]
        target_node = neg[1]
        for i,c in enumerate(colours):
            node = f"n_{k}_{i}"
            nodes = pd.concat([nodes, pd.DataFrame(columns=["node_id", "color"], data=[{"node_id": node, "color": c}])], ignore_index=True)
        for i in range(len(edge)):
            for j in range(len(edge)):
                if edge[i][j] == 1:
                    edges = pd.concat([edges, pd.DataFrame(columns=["node_1", "node_2"], data=[{"node_1": f"n_{k}_{i}", "node_2": f"n_{k}_{j}"}])], ignore_index=True)
        cyclic = pd.concat([cyclic, pd.DataFrame(columns=["id","class"], data=[{"id": target_node, "class": "neg"}])], ignore_index=True)

    nodes.to_csv(f"{output_path}/nodes.csv", index=False)
    edges.to_csv(f"{output_path}/edges.csv", index=False)
    if dataset_name is not None:
        cyclic.to_csv(f"{output_path}/{dataset_name}.csv", index=False)
    else:
        cyclic.to_csv(f"{output_path}/cyclic.csv", index=False)

    return


# if __name__ == "__main__":
#     generate_relational_data(output_path="docker/Benchmark/cyclic/relational",min_graph=11,max_graph=11,num_graphs=200,cycle_size=2)