import numpy as np

# Gini Impurity for a set of labels
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

# Entropy (Shannon) for a set of labels
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # filter out zero probs to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))
