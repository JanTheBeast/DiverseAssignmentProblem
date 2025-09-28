import algorithm
import generate_data
import numpy as np

def array_to_string(A):
    res = ""
    for row in A:
        for col in row:
            res += str(col) + " "
        res += "\n"
    return res

def write_file(name, n, G, D):
    f = open(name, "w")
    min_div = algorithm.get_minimum_diversity(D, n)
    max_div = algorithm.get_maximum_diversity(D, n)
    min_cost = algorithm.get_minimum_cost(G, n)
    max_cost = algorithm.get_maximum_cost(G, n)

    f.write(str(n) + " " + str(min_div) + " " + str(max_div) + " " + str(min_cost) + " " + str(max_cost) + "\n")
    f.write(array_to_string(G))
    f.write(array_to_string(D).removesuffix("\n"))
    f.close()

def write_points(name, points):
    f = open(name, "w")
    f.write(str(len(points)))
    for point in points:
        f.write("\n")
        f.write(str(point[0]) + " " + str(point[1]))
    f.close()

def read_points(name):
    f = open(name, "r")
    n = int(f.readline())
    f.close()
    full_arr = np.loadtxt(name, skiprows=1)
    result = []
    if n == 1:
        result += [(full_arr[0], full_arr[1])]
    else:
        for arr in full_arr:
            result += [(arr[0], arr[1])]
    return n, result

def read_timings(name):
    full_arr = np.loadtxt(name)
    return full_arr.T

def create_empty_file(name):
    f = open(name, "w")
    f.close()

def file_append_num(name, num):
    f = open(name, "a")
    f.write(str(num) + "\n")
    f.close()

def generate_datasets(sizes=[4,8,16,32,64,128]):
    for size in sizes:
        for i in range(10):
            G = generate_data.generate_data(size)
            D = generate_data.generate_random_diversity(size)
            write_file("dat/random_div_"+ str(size) + "_" + str(i), size, G, D)
            G = generate_data.generate_data(size)
            D = generate_data.generate_disjoint_diversity(size)
            write_file("dat/disjoint_div_"+ str(size) + "_" + str(i), size, G, D)
            G = generate_data.generate_data(size)
            D = generate_data.generate_diversiy_by_distance(size)
            write_file("dat/distance_div_"+ str(size) + "_" + str(i), size, G, D)
            G = generate_data.generate_data(size)
            D = generate_data.generate_uniform_diversity(size)
            write_file("dat/uniform_div_"+ str(size) + "_" + str(i), size, G, D)

def load_file(file_name):
    f = open(file_name, "r")
    [n, min_div, max_div, min_cost, max_cost] = f.readline().split()
    f.close()
    full_arr = np.split(np.loadtxt(file_name, skiprows=1), [int(n)])
    G = full_arr[0]
    D = full_arr[1]
    return int(n), float(min_div), float(max_div), float(min_cost), float(max_cost), G, D