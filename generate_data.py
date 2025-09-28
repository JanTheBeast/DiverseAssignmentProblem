### File that was used for generating the datasets

import numpy as np

# Generate matrix of bipartite graph
def generate_data(n):
    return np.random.random_integers(1, 100, (n, n))

# Generate diversity graph with value -1 and 1
def generate_uniform_diversity(n):
    matrix = np.random.random_integers(0, 1, (n,n))
    matrix *= 2
    matrix -= 1
    for i in range(n):
        for j in range(i+1, n):
            matrix[j][i] = matrix[i][j]
    return matrix

# Generate diversity with real numbers
def generate_random_diversity(n):
    matrix = np.random.random((n,n))
    matrix = (matrix + matrix.T) / 2
    return matrix

# Generate diversity graph with 0 on self-loops, 1 otherwise
def generate_disjoint_diversity(n):
    matrix = np.ones((n,n)) - np.identity(n)
    return matrix

def generate_diversiy_by_distance(n):
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = min((i-j+n) % n, (j-i+n) % n)
    return matrix
