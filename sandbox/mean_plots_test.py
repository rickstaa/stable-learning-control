import numpy as np

# TODO: Use only longest lenghts
test_array = np.array([[0, 1, 2, 3], [2, 3, 4], [3, 4], [4, 5, 6, 7]], dtype=object)
mean = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]).mean(axis=1)
mean = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]]).std(axis=1)
print(test_array)