import exact, subroutines, fileio, algorithm
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Calculate n+! points using exact algorithm
# We do only n+1 points because so does the heuristic
def get_exact_points(G, D, n):
    result = []
    for i in range(n+1):
        ass, cost, div = exact.solve_ip(G, D, n, i/n)
        result += [(cost, div)]
    dominating_set = subroutines.get_dominating_set(result)
    return dominating_set

# Timing test for exact and approximate algorithm
def run_timing_test(divs, sizes):
    inst = 0
    for size in sizes:
        for div in divs:
            for i in range(10):
                inst += 1
                inst_name = div + "_"+ str(size) + "_" + str(i)
                print("\rRunning timing test: " + inst_name + " (" + str(inst) + "/" + str(10 * len(sizes) * len(divs)) + ")", end='', flush=True)
                (n, G, D) = fileio.load_file("data/" + inst_name)
                start = time.time()
                algorithm.get_algorithm_points(G, D, n)
                end = time.time()
                fileio.file_append_num("timing/approx/" + str(size), end - start)
                start = time.time()
                get_exact_points(G, D, n)
                end = time.time()
                fileio.file_append_num("timing/exact/" + str(size), end - start)
    print()

# Make plots of the running times in regular scale and log scale
def plot_timings(sizes):
    arr_approx = []
    arr_exact = []
    std_approx = []
    std_exact = []
    for size in sizes:
        timings = fileio.read_timings("timing/approx/" + str(size))
        arr_approx += [np.average(timings)]
        timings = fileio.read_timings("timing/exact/" + str(size))
        arr_exact += [np.average(timings)]
        std_approx += [np.std(timings)]
        std_exact += [np.std(timings)]
    plt.errorbar(sizes, arr_approx, std_approx, linestyle='None', marker='^')
    plt.errorbar(sizes, arr_exact, std_exact, linestyle='None', marker='^')
    fig, axs = plt.subplots(ncols=2)
    axs[1].set_xscale("log", base=2)
    axs[1].set_yscale("log", base=10)
    for ax in axs:
        ax.set_xlabel("Size of instance")
        ax.set_ylabel("Running time (s)")
        ax.set_xticks(sizes)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.3g}'.format(y)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.3g}'.format(x)))
        ax.plot(sizes, arr_approx, marker='o',label="Approximation")
        ax.plot(sizes, arr_exact, marker='o',label="Exact")
        ax.legend(loc="upper left")
    fig.set_figheight(4.8 * 1.5)
    fig.set_figwidth(2 * 6.4 * 1.5)
    plt.show()