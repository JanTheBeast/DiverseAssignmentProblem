### File containing routines for solving subproblems used in the heuristic. 
### It contains cardinality-constrained matching, the (unbalanced) transportation problem
### and getting the dominating set from a set of points.

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# Solves the k-cardinality constrained 2-matching problem
def solve_k_card_2_matching(matrix, k):

    n = len(matrix)
    total = np.sum(np.abs(matrix)) + 1 # Arbitrary big number

    # Initialize bipartite graph with 4n - 2k nodes
    B = nx.Graph()
    B.add_nodes_from(list(range(n)), bipartite=0)
    B.add_nodes_from(list(range(n,2*n)), bipartite=1)
    B.add_nodes_from([x + 2 * n for x in list(range(n-k))], bipartite=0)
    B.add_nodes_from([x + 3 * n - k for x in list(range(n-k))], bipartite=1)

    edges = []

    # Add edges (u,v') and (v,u') for (u,v) in G to create assignment problem
    for i in range(n):
        for j in range(n):
            edges += [(i, j+n, -matrix[i][j]), (j, i+n, -matrix[i][j])]

    # Add low weight edges between dummy nodes
    for i in range(n-k):
        for j in range(n-k):
            edges += [(i + 2*n, j + 3*n-k, total)]

    # Between dummy nodes and real nodes add high weight edges (forced to pick)
    for i in range(n-k):
        for j in range(n):
            edges += [(i + 2*n, j + n, -total), (j, i + 3*n-k, -total)]

    B.add_weighted_edges_from(edges)

    # Solve min-weight matching with inverse weights, so max-weight matching
    matching = nx.bipartite.minimum_weight_full_matching(B)

    # Create a dictionary of edges based on the resulting matching, where value is multiplicity
    result = {}
    for v1, v2 in matching.items():
        if v1 >= n or v2 >= 2*n:
            continue
        v2 -= n
        if v1 <= v2:
            result[(v1,v2)] = result.get((v1,v2), 0) + 1
        else:
            result[(v2,v1)] = result.get((v2,v1), 0) + 1

    return result

# Solves the (unbalanced) transportation problem with certain supplies and demands
def solve_transportation(matrix, supplies, demands):
    n1 = len(matrix)
    n2 = len(matrix[0])

    # Create flow graph with source and sink
    G = nx.DiGraph()
    G.add_nodes_from(list(range(n1 + n2 + 2)))

    s = n1 + n2
    t = n1 + n2 + 1

    edges = []

    # Add source edges
    for i in range(n1):
        edges += [(s, i, {"capacity": int(supplies[i]), "weight": 0})]

    # Add sink edges
    for i in range(n2):
        edges += [(i+n1, t, {"capacity": int(demands[i]), "weight": 0})]

    # Add existing edges
    for i in range(n1):
        for j in range(n2):
            # Use negative weights to get max cost
            edges += [(i, j+n1, {"capacity": 2, "weight": -int(matrix[i][j])})]

    G.add_edges_from(edges)

    # Calculate flow
    flow = nx.max_flow_min_cost(G, s, t)
    return flow

# Get the dominating set from a set of points
def get_dominating_set(points):
    points.sort(reverse=True)
    max_y = points[0][1]
    dominating_set = [points[0]]
    for point in points[1:]:
        if point[1] > max_y:
            max_y = point[1]
            dominating_set += [point]
    return dominating_set

# Draw a scatter plot
def plot_points(points1, points2):
    plt.scatter(*zip(*points1))
    plt.scatter(*zip(*points2), color='red')
    plt.xlabel("Matching weights")
    plt.ylabel("Diversity")
    plt.show()

# Draw a graph in plt
def draw_graph(G):
    top = nx.bipartite.sets(G)[0]
    pos = nx.bipartite_layout(G, top)
    nx.draw(G, pos=pos, with_labels = True)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels,label_pos=0.3)
    plt.show()