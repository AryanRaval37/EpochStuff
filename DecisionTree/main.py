import numpy as np
from DecisionTree import DecisionTree
from impurity_metrics import *

## Almost same as the KNN structure

data = np.array([
    [12.0, 1.5, 1, 'Wine'],
    [5.0, 2.0, 0, 'Beer'],
    [40.0, 0.0, 1, 'Whiskey'],
    [13.5, 1.2, 1, 'Wine'],
    [4.5, 1.8, 0, 'Beer'],
    [38.0, 0.1, 1, 'Whiskey'],
    [11.5, 1.7, 1, 'Wine'],
    [5.5, 2.3, 0, 'Beer']
])

# Splitting into labels and features
X = data[:, :-1].astype(dtype=float)  # All columns except the last one (features)
y = data[:, -1]   # Last column (labels)

# Step 1 : label encoding (one hot encoding) - pretty sure this is not how its done
# Wine    = 1
# Beer    = 2
# Whiskey = 4
label_map = {'Wine': 1, 'Beer': 2, 'Whiskey': 4}
reverse_label_map = {v: k for k, v in label_map.items()}
y = np.array([label_map[label] for label in y])


print("Shape of data is ", data.shape)
print("Shape of features (X):", X.shape)
print("Shape of labels (y):", y.shape)
print("Features:\n", X)
print("Labels:\n", y)

# build the decision tree with gini impurity
myDT = DecisionTree(max_depth=5, min_samples_split=2, impurity_metric=gini)
myDT.fit(X, y)

# print the tree structure
feature_names = ['Alcohol', 'Sugar', 'Color']
print("\n--- Decision Tree Structure ---")
myDT.print_tree(feature_names=feature_names)

# test predictions
test_data = np.array([
    [6.0, 2.1, 0],   # Expected: Beer
    [39.0, 0.05, 1], # Expected: Whiskey
    [13.0, 1.3, 1]   # Expected: Wine
], dtype=float)

predictions = myDT.predict(test_data)
print("\n--- Predictions ---")
for i in range(len(test_data)):
    print(f"Input: {test_data[i]}  ->  Predicted: {reverse_label_map[predictions[i]]}")

# NOTE: here the dataset is simple enough to ignore out parameters and just make the tree based on percent alcohol
