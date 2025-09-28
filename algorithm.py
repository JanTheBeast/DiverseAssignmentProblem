import generate_data
import subroutines
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

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

# Generate two matchings from a 2-matching
def split_matchings(assignment):
    n = len(assignment)

    # Create graph and add 2n nodes
    B = nx.Graph()
    B.add_nodes_from(list(range(2*n)))

    # Add edges that occur in the 2-matching
    edges = []
    for i in range(n):
        for j in range(n):
            if assignment[i][j] > 0:
                edges += [(i, j+n)]
    B.add_edges_from(edges)

    # Find one of the two matchings
    matching = nx.algorithms.maximal_matching(B)
    M1 = []
    M2 = []
    for i in range(n):
        for j in range(n):

            # Skip unassigned edges
            if assignment[i][j] == 0:
                continue

            # If multiplicity is 2 add to both matchings
            if assignment[i][j] == 2:
                M1 += [(i, j)]
                M2 += [(i, j)]

            # If mulitplicity is 1 add to one matching
            elif (i,j+n) in matching:
                M1 += [(i, j)]
            else:
                M2 += [(i, j)]
    return (M1, M2)

# Gets the optimal 2-matching in a diversity graph
def get_maximum_diversity(D, n):

    # Solve 2-matching
    two_matching = subroutines.solve_k_card_2_matching(D, n)

    # Calculate diversity
    diversity = 0
    for i in range(n):
        for j in range(i,n):
            if two_matching.get((i,j), 0) > 0:
                diversity += D[i][j] * int(two_matching[(i,j)])

    return diversity

# Gets min-weight 2-matching in diversity graph
def get_minimum_diversity(D, n):

    # Calculate optimal 2-matching with inverse weights
    D = -D
    two_matching = subroutines.solve_k_card_2_matching(D, n)
    D = -D

    # Calculate diversity
    diversity = 0
    for i in range(n):
        for j in range(i,n):
            if two_matching.get((i,j), 0) > 0:
                diversity += D[i][j] * int(two_matching[(i,j)])

    return diversity

# Gets the maximal weight of 2 matchings in G
def get_maximum_cost(G, n):

    # Solve transportation
    supplies = 2 * np.ones(n)
    demands = 2 * np.ones(n)
    flow = subroutines.solve_transportation(G, supplies, demands)

    # Calculate weight of matchings
    cost = 0
    for i in range(n):
        for j in range(n,2*n):
            cost += flow[i][j] * G[i][j-n]

    return cost

# Gets the maximal weight of 2 matchings in G
def get_minimum_cost(G, n):

    # Solve transportation
    supplies = 2 * np.ones(n)
    demands = 2 * np.ones(n)
    flow = subroutines.solve_transportation(-G, supplies, demands)

    # Calculate weight of matchings
    cost = 0
    for i in range(n):
        for j in range(n,2*n):
            cost += flow[i][j] * G[i][j-n]

    return cost

# Runs our algorithm in a (G, D, k) instance
def run_algorithm(G, D, n, k):

    # Solve k-cardinality 2-matching
    two_matching = subroutines.solve_k_card_2_matching(D, k)

    # Make a transportation from 2-matching to L
    H = np.zeros((n, n*n))
    supplies = np.zeros(n)
    demands = np.zeros(n*n)

    # Create auxiliary graph weights and demands
    for i in range(n):
        supplies[i] = 1 # Supply is always 1
        for j in range(n):
            for k in range(j,n):

                # Set weight of edges
                H[i][j*n + k] = G[i][j] + G[i][k]

                # Set demand to demand = x_e
                demands[j*n + k] = two_matching.get((j,k), 0)

    flow = subroutines.solve_transportation(H, supplies, demands) 

    # Create a matching from the transportation
    edge_matching = []
    for i in range(n):
        for j in range(n, n*n+n):
            if flow[i][j] > 0:
                edge_matching += [(i,((j-n) // n, (j-n) % n), flow[i][j])]

    # Matching remaining nodes using a transportation
    supplies = 2 * np.ones(n)
    demands = 2 * np.ones(n)

    assignment = np.zeros((n,n))
    for match in edge_matching:

        # Set supply = 
        supplies[match[0]] = 0

        # Set demand = 2 - x_e 
        demands[match[1][0]] -= match[2]
        demands[match[1][1]] -= match[2]

        # Assign nodes based on the earlier matching
        assignment[match[0]][match[1][0]] += match[2]
        assignment[match[0]][match[1][1]] += match[2]

    flow = subroutines.solve_transportation(G, supplies, demands)

    # Make matching based on transportation
    matching = []
    for i in range(n):
        for j in range(n, 2*n):
            if flow[i][j] > 0:
                matching += [(i,j-n,flow[i][j])]

    # Make assignment based on matching
    for match in matching:
        assignment[match[0]][match[1]] += match[2]

    # Calculate cost of solution
    cost = 0
    for i in range(n):
        for j in range(n):
            cost += assignment[i][j] * G[i][j]

    # Calculate diversity of solution
    diversity = 0
    for i in range(n):
        r = []
        for j in range(n):
            if assignment[i][j] > 0:
                r += [j] * int(assignment[i][j])
        diversity += D[r[0]][r[1]]

    return assignment, cost, diversity
