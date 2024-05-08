import numpy as np
import timeit
import matplotlib.pyplot as plt
import itertools

# * Define the three functions

# * Method 1: with a for loop
def cs_for(x):
    xc = []
    for i in range(len(x)):
        if i == 0:
            xc.append(x[i])
        else:
            xc.append(sum(x[:i+1]))
    return xc

# * Method 2: with itertools
def cumsum_it(x):
    return list(itertools.accumulate(x))

# * Method 3: numpy cumsum
def np_cumsum(x):
    return np.cumsum(x)

# * Measure execution time for different sizes of x
sizes = range(1, 101)
times_cs_for = []
times_cumsum_it = []
times_np_cumsum = []

for size in sizes:
    x = np.random.rand(size)
    time_for = timeit.timeit(lambda: cs_for(x), number=100)
    times_cs_for.append(time_for)
    
    time_it = timeit.timeit(lambda: cumsum_it(x), number=100)
    times_cumsum_it.append(time_it)
    
    time_np = timeit.timeit(lambda: np_cumsum(x), number=100)
    times_np_cumsum.append(time_np)

# Plot the mean execution time of each function
plt.plot(sizes, times_cs_for, label='cs_for')
plt.plot(sizes, times_cumsum_it, label='cumsum_itertools')
plt.plot(sizes, times_np_cumsum, label='np_cumsum')
plt.xlabel('Size of x')
plt.ylabel('Mean Execution Time (s)')
plt.title('Mean Execution Time of Cumulative Sum Functions')
plt.legend()
plt.show()
