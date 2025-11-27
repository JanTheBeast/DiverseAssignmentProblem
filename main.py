import generate_data
import timing_test, pareto_tests
import exact, algorithm,fileio

exact.setup_env()

sizes = [4, 8, 16, 32, 64, 128]
generate_data.generate_datasets(sizes) # Comment out if already generated

divs = ["disjoint_div", "distance_div", "uniform_div", "random_div"]

pareto_tests.run_test(divs, sizes) # Comment out if already run
pareto_tests.calculate_pareto_stats(divs, sizes)

print()

timing_test.run_timing_test(divs, sizes) # Comment out if already run
timing_test.plot_timings(sizes)

print()