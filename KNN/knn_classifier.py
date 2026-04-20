import numpy as np
from KNN import KNN
from distance_metrics import *

data = np.array([
    [150, 7.0, 1, 'Apple'],
    [120, 6.5, 0, 'Banana'],
    [180, 7.5, 2, 'Orange'],
    [155, 7.2, 1, 'Apple'],
    [110, 6.0, 0, 'Banana'],
    [190, 7.8, 2, 'Orange'],
    [145, 7.1, 1, 'Apple'],
    [115, 6.3, 0, 'Banana']
])

# Splitting into labels and features
X = data[:, :-1].astype(dtype=float)  # All columns except the last one (features)
y = data[:, -1]   # Last column (labels)

# Step 1 : label encoding - one hot encoding
# Apple  - 100   = 4
# Banana - 010   = 2
# Orange - 001   = 1
# Note: Helpful for some reason... remember to think about why

# Look into np.eye - probably faster for one hot encoding
label_map = {'Apple': 1, 'Banana': 2, 'Orange': 4}
y = np.array([label_map[label] for label in y])


print("Shape of data is ", data.shape)
print("Shape of features (X):", X.shape)
print("Shape of labels (y):", y.shape)
print("Features:\n", X)
print("Labels:\n", y)

myKNN = KNN(k=3, distance_metric=euclidean)
test_data = np.array([
    [118, 6.2, 0],  # Expected: Banana
    [160, 7.3, 1],  # Expected: Apple
    [185, 7.7, 2]   # Expected: Orange
], dtype=float)
myKNN.fit(X, y)
predictions = myKNN.predict(test_data)
print(predictions)


    












