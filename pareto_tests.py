import fileio, exact, algorithm

# Calculate pareto front for given instances
def preprocess_pareto(divs, sizes):
    inst = 0
    for size in sizes:
        for div in divs:
            for i in range(10):
                inst += 1
                inst_name = div + "_"+ str(size) + "_" + str(i)
                print("\rCalculating exact solution: " + inst_name + " (" + str(inst) + "/" + str(10 * len(sizes) * len(divs)) + ")", end='', flush=True)                
                (n, G, D) = fileio.load_file("data/" + inst_name)
                pareto = exact.get_pareto_front(G, D, n)
                fileio.write_points("pareto/exact/" + inst_name, pareto)
    print()

# Calculate approximate solutions for given instances
def preprocess_approx(divs, sizes):
    inst = 0
    for size in sizes:
        for div in divs:
            for i in range(10):
                inst += 1
                inst_name = div + "_"+ str(size) + "_" + str(i)
                print("\rCalculating approximate solution: " + inst_name + " (" + str(inst) + "/" + str(10 * len(sizes) * len(divs)) + ")", end='', flush=True)
                (n, G, D) = fileio.load_file("data/" + inst_name)
                pareto = algorithm.get_algorithm_points(G, D, n)
                fileio.write_points("pareto/approx/" + inst_name, pareto)
    print()

def run_test(divs, sizes):
    preprocess_approx(divs, sizes)
    preprocess_pareto(divs, sizes)

# Function for calculating area of (approximate) pareto front
def calculate_set_area(points, min_div, min_cost):
    points.sort()
    result = (points[0][0] - min_cost) * (points[0][1] - min_div)
    for i in range(1, len(points)):
        result += (points[i][0] - points[i-1][0]) * (points[i][1] - min_div)
    return result

# Calculate summary of stats
def calculate_pareto_stats(divs, sizes):
    pareto_fraction = []
    points_approx = []
    points_exact = []
    inst = 0
    for div in divs:
        total_frac = []
        total_points_approx = []
        total_points_exact = []
        for size in sizes:
            total_frac += [0]
            total_points_approx += [0]
            total_points_exact += [0]
            for i in range(10):
                inst += 1
                inst_name = div + "_"+ str(size) + "_" + str(i)
                print("\rCalculating statistics: " + inst_name + " (" + str(inst) + "/" + str(10 * len(sizes) * len(divs)) + ")", end='', flush=True)
                approx = fileio.read_points("pareto/approx/" + inst_name)
                exact = fileio.read_points("pareto/exact/" + inst_name)

                (n, G, D) = fileio.load_file("data/" + inst_name)
                min_cost = algorithm.get_minimum_cost(G, n)
                min_div = algorithm.get_minimum_diversity(D, n)

                frac1 = calculate_set_area(approx[1], min_div, min_cost)
                frac2 = calculate_set_area(exact[1], min_div, min_cost)

                total_frac[-1] += frac1 / frac2
                total_points_approx[-1] += approx[0]                
                total_points_exact[-1] += exact[0]                
            total_frac[-1] /= 10
            total_points_approx[-1] /= 10
            total_points_exact[-1] /= 10
        pareto_fraction += [total_frac]
        points_approx += [total_points_approx]
        points_exact += [total_points_exact]
    fileio.write_array("pareto/stats/pareto_fraction", pareto_fraction)
    fileio.write_array("pareto/stats/points_approximation", points_approx)
    fileio.write_array("pareto/stats/points_exact", points_exact)
    print()
