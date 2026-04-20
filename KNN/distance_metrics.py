import numpy as np

# Eucliedian distance between two points (vectors)
def euclidean(a, b):
    return np.sqrt((b-a)@(b-a))

def manhattan(a, b):
    return np.abs(a-b)

def minkowski(a, b, p=3):
    return np.power(np.sum(np.power(np.abs(a-b), p)), 1/p)
