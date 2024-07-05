import numpy as np
import matplotlib.pyplot as plt


def create_random_array():
    return np.random.rand(5, 5)


def get_random_index(array):
    i = np.random.randint(0, len(array))
    j = np.random.randint(0, len(array))
    return i, j


results = {
    "raw": [],
    "mean replaced": [],
    "median replaced": [],
    "nan replaced": [],
}
for i in range(20):
    arr = create_random_array()
    idx = get_random_index(arr)

    results["raw"].append(arr.mean())

    arr_mean = arr.copy()
    arr_mean[idx] = np.mean(arr)
    results["mean replaced"].append(arr_mean.mean())

    arr_med = arr.copy()
    arr_med[idx] = np.median(arr_med)
    results["median replaced"].append(arr_med.mean())

    arr_nan = arr.copy()
    arr_nan[idx] = np.nan
    results["nan replaced"].append(np.nanmean(arr_nan))

print(results)

plt.plot(results["raw"], label="Raw")
plt.plot(results["mean replaced"], label="Mean Replaced")
plt.plot(results["median replaced"], label="Median Replaced")
plt.plot(results["nan replaced"], label="Nan Replaced")
plt.xlabel("Iteration")
# plt.ylabel("Sum")
plt.legend()
plt.show()
