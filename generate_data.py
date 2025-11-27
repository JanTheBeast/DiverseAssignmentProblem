### File that was used for generating the datasets

import numpy as np
import fileio

# Generate matrix of bipartite graph with random weights
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
    matrix = np.random.random_integers(1, 100, (n,n))
    matrix = matrix + matrix.T
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

def generate_datasets(sizes=[4,8,16,32,64,128]):
    for size in sizes:
        for i in range(10):
            G = generate_data(size)
            D = generate_random_diversity(size)
            fileio.write_file("data/random_div_"+ str(size) + "_" + str(i), size, G, D)
            G = generate_data(size)
            D = generate_disjoint_diversity(size)
            fileio.write_file("data/disjoint_div_"+ str(size) + "_" + str(i), size, G, D)
            G = generate_data(size)
            D = generate_diversiy_by_distance(size)
            fileio.write_file("data/distance_div_"+ str(size) + "_" + str(i), size, G, D)
            G = generate_data(size)
            D = generate_uniform_diversity(size)
            fileio.write_file("data/uniform_div_"+ str(size) + "_" + str(i), size, G, D)
