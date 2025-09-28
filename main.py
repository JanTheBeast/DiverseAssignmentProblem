import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np
import algorithm
import exact
import time
import subroutines
import fileio
import gurobipy as gp

# Get the dominating set from a set of poitns
def get_dominating_set(points):
    points.sort(reverse=True)
    max_y = points[0][1]
    dominating_set = [points[0]]
    for point in points[1:]:
        if point[1] > max_y:
            max_y = point[1]
            dominating_set += [point]
    return dominating_set

# Calculate n points using exact algorithm, used for timing test
def get_exact_points(G, D, n, min_div, max_div):
    result = []
    for i in range(n+1):
        ass, cost, div = exact.solve_ip(G, D, n, min_div + (i / n) * (max_div - min_div))
        result += [(cost, div)]
    dominating_set = get_dominating_set(result)
    return dominating_set

# Calculate n points using the approximate algorithm
def get_algorithm_points(G, D, n):
    result = []
    for i in range(n+1):
        ass, cost, div = algorithm.run_algorithm(G, D, n, i)
        result += [(cost, div)]
    dominating_set = get_dominating_set(result)
    return dominating_set

# Recursively calculate pareto front using exact algorithm
def get_pareto_front_recursive(G, D, n, start, end, depth):
    result = []
    if depth <= 10 and start[0] > end[0] + 1e-4:
        ass, cost, div = exact.solve_ip(G, D, n, (start[1] + end[1]) / 2)
        print((cost,div))
        if not (cost == end[0] and div == end[1]):
            list1 = get_pareto_front_recursive(G, D, n, start, (cost, div), depth+1)
            list2 = get_pareto_front_recursive(G, D, n, (cost, div), end, depth+1)
        else:
            list1 = [start]
            list2 = []
        result += list1 + list2
    else:
        result = [start]
    return result

# Calling function for recursive process
def get_pareto_front(G, D, n, min_div, max_div, min_cost, max_cost):
    result = get_pareto_front_recursive(G, D, n, (max_cost, min_div), (min_cost, max_div), 0)
    result += [(min_cost, max_div)]
    dominating_set = get_dominating_set(result)
    return dominating_set

# Function for calculating area of (approximate) pareto front
def calculate_set_area(points, min_div, min_cost):
    points.sort()
    result = (points[0][0] - min_cost) * (points[0][1] - min_div)
    for i in range(1, len(points)):
        result += (points[i][0] - points[i-1][0]) * (points[i][1] - min_div)
    return result

# Calculate algorithm performance using exact and approximate pareto fronts 
def get_approximation_fraction(G, D, n, min_div, max_div, min_cost, max_cost):
    algo_points = get_algorithm_points(G, D, n)
    par_points = get_pareto_front(G, D, n, min_div, max_div, min_cost, max_cost)
    frac1 = calculate_set_area(algo_points, min_div, max_div, min_cost, max_cost)
    frac2 = calculate_set_area(par_points, min_div, max_div, min_cost, max_cost)
    return frac1 / frac2

# Calculate pareto front for given instances
def preprocess_pareto(divs, sizes):
    for size in sizes:
        for div in divs:
            for i in range(10):
                inst_name = div + "_"+ str(size) + "_" + str(i)
                (n, min_div, max_div, min_cost, max_cost, G, D) = fileio.load_file("data/" + inst_name)
                print("\n" + inst_name)
                pareto = get_pareto_front(G, D, n, min_div, max_div, min_cost, max_cost)
                fileio.write_points("exact/" + inst_name, pareto)

# Calculate approximate solutions for given instances
def preprocess_approx(divs, sizes):
    for size in sizes:
        for div in divs:
            for i in range(10):
                inst_name = div + "_"+ str(size) + "_" + str(i)
                (n, min_div, max_div, min_cost, max_cost, G, D) = fileio.load_file("data/" + inst_name)
                print("\n" + inst_name)
                pareto = get_algorithm_points(G, D, n)
                fileio.write_points("approx/" + inst_name, pareto)

# Timing test for exact and approximate algorithm
def run_timing_test(divs, sizes):
    for size in sizes:
        for i in range(10):
            for div in divs:
                inst_name = div + "_"+ str(size) + "_" + str(i)
                print(inst_name)
                (n, min_div, max_div, min_cost, max_cost, G, D) = fileio.load_file("data/" + inst_name)
                start = time.time()
                get_algorithm_points(G, D, n)
                end = time.time()
                fileio.file_append_num("timing/approx/" + str(size), end - start)
                start = time.time()
                get_exact_points(G, D, n, min_div, max_div)
                end = time.time()
                fileio.file_append_num("timing/exact/" + str(size), end - start)

# Calculate summary of stats
def calculate_pareto_stats(divs, sizes):
    for size in sizes:
        for div in divs:
            print("\nSize = " + str(size) + ", Div = " + div + ": ")
            total_frac = 0
            total_points_approx = 0
            total_points_exact = 0
            for i in range(10):
                inst_name = div + "_"+ str(size) + "_" + str(i)
                (n, min_div, max_div, min_cost, max_cost, G, D) = fileio.load_file("data/" + inst_name)
                approx = fileio.read_points("approx/" + inst_name)
                exact = fileio.read_points("exact/" + inst_name)
                frac1 = calculate_set_area(approx[1], min_div, min_cost)
                frac2 = calculate_set_area(exact[1], min_div, min_cost)
                total_frac += frac1 / frac2
                total_points_approx += approx[0]                
                total_points_exact += exact[0]                
            total_frac /= 10
            total_points_approx /= 10
            total_points_exact /= 10
            print("Average Pareto fraction: " + str(total_frac))
            print("Average points approximate: " + str(total_points_approx))
            print("Average points exact: " + str(total_points_exact))

# Make plots
def plot_timings(sizes):
    arr_approx = []
    arr_exact = []
    std_approx = []
    std_exact = []
    for size in sizes:
        timings = fileio.read_timings("timing/approx/" + str(size))
        arr_approx += [np.average(timings)]
        std_approx += [np.std(timings)]
        timings = fileio.read_timings("timing/exact/" + str(size))
        arr_exact += [np.average(timings)]
        std_exact += [np.std(timings)]
    # plt.errorbar(sizes, arr_approx, std_approx, linestyle='None', marker='^')
    # plt.errorbar(sizes, arr_exact, std_exact, linestyle='None', marker='^')
    fig, ax = plt.subplots()
    ax.set_xlabel("Size of instance")
    ax.set_ylabel("Running time (s)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(sizes)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.3g}'.format(y)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.3g}'.format(x)))
    ax.plot(sizes, arr_approx, marker='o',label="Approximation")
    ax.plot(sizes, arr_exact, marker='o',label="Exact")
    ax.legend(loc="upper left")
    fig.set_figheight(4.8 * 1.5)
    fig.set_figwidth(6.4 * 1.5)
    plt.show()

# divs = ["disjoint_div", "uniform_div", "random_div", "distance_div"]
# sizes = [4, 8, 16, 32, 64]
# calculate_pareto_stats(divs, sizes)

divs = ["disjoint_div", "uniform_div", "random_div", "distance_div"]
sizes = [4,8,16,32]
preprocess_approx(divs, sizes)

# divs = ["random_div", "distance_div"]
# sizes = [64]
# preprocess_pareto(divs, sizes)

# divs = ["distance_div", "disjoint_div", "uniform_div"]
# sizes = [4, 4, 8, 128]
# run_timing_test(divs, sizes)

# preprocess_pareto(divs, sizes)
# (n, min_div, max_div, min_cost, max_cost, G, D) = fileio.load_file("data/random_div_16_8")
# get_approximation_fraction(G, D, n, min_div, max_div, min_cost, max_cost)

# start = time.time()
# algorithm.run_algorithm(G, D, n, n)
# end = time.time()
# print(end - start)
# min_div = algorithm.get_minimum_diversity(D, n)
# max_div = algorithm.get_maximum_diversity(D, n)
# min_cost = algorithm.get_minimum_cost(G, n)
# max_cost = algorithm.get_maximum_cost(G, n)

# print(load_file("data/random_div_4_0"))
# run_timing_tests()

sizes = [4,8,16,32,64,128]
plot_timings(sizes)