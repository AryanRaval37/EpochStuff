import numpy as np
from impurity_metrics import gini

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # if leaf node: majority class

class DecisionTree:
    # expects max_depth, min_samples_split, and impurity metric
    def __init__(self, max_depth=10, min_samples_split=2, impurity_metric=gini):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.impurity = impurity_metric
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # stopping conditions
        # 1. all labels are the same
        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        # 2. max depth reached
        if depth >= self.max_depth:
            return Node(value=self._majority_class(y))
        # 3. not enough samples to split
        if n_samples < self.min_samples_split:
            return Node(value=self._majority_class(y))

        # find the best split
        best_feature, best_threshold = self._best_split(X, y, n_features)

        # if no valid split found, make a leaf
        if best_feature is None:
            return Node(value=self._majority_class(y))

        # split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # recursively build left and right subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)
    
    def _best_split(self, X, y, n_features):
        best_impurity = float('inf')
        best_feature = None
        best_threshold = None

        # loop over all features
        for feature_idx in range(n_features):
            # get all unique values for this feature as possible thresholds
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # NOTE : remember this 'masking' technique next time...
                # split into left and right
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # skip if one side is empty
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # compute weighted impurity
                left_impurity = self.impurity(y[left_mask])
                right_impurity = self.impurity(y[right_mask])
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                weighted_impurity = (n_left * left_impurity + n_right * right_impurity) / len(y)

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _majority_class(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict_one(self, x):
        # traverse the tree from root
        node = self.root
        while node.value is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X_test):
        return [self.predict_one(X_test[i]) for i in range(len(X_test))]


    # ! AI generated debug function
    def print_tree(self, node=None, depth=0, feature_names=None):
        if node is None:
            node = self.root

        indent = "  " * depth

        # leaf node
        if node.value is not None:
            print(f"{indent}-> Class: {node.value}")
            return

        # internal node
        feat_name = feature_names[node.feature_index] if feature_names else f"Feature[{node.feature_index}]"
        print(f"{indent}if {feat_name} <= {node.threshold}:")
        self.print_tree(node.left, depth + 1, feature_names)
        print(f"{indent}else ({feat_name} > {node.threshold}):")
        self.print_tree(node.right, depth + 1, feature_names)
