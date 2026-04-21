### Visualising KNN
# ! Parts of the file are AI generated...

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN
from distance_metrics import euclidean

# ── Data ──────────────────────────────────────────────────────────────
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

X = data[:, :-1].astype(float)
y_raw = data[:, -1]

label_map = {'Apple': 1, 'Banana': 2, 'Orange': 4}
reverse_label_map = {v: k for k, v in label_map.items()}
y = np.array([label_map[label] for label in y_raw])

# ── Pick two features to plot ─────────────────────────────────────────
# Feature 0: Weight,  Feature 1: Sweetness  (most visually separable)
feat_x, feat_y = 0, 1
feature_names = ['Weight', 'Sweetness', 'Color Code']

X_2d = X[:, [feat_x, feat_y]]

# ── Fit KNN on the 2-feature subset ──────────────────────────────────
k = 3
knn = KNN(k=k, distance_metric=euclidean)
knn.fit(X_2d, y)

# ── Build decision boundary mesh ─────────────────────────────────────
h = 0.5  # step size
x_min, x_max = X_2d[:, 0].min() - 10, X_2d[:, 0].max() + 10
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))

mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = np.array(knn.predict(mesh_points))
Z = Z.reshape(xx.shape)

# ── Colours ───────────────────────────────────────────────────────────
# Map encoded labels (1, 2, 4) → indices (0, 1, 2) for colourmaps
label_to_idx = {1: 0, 2: 1, 4: 2}
Z_idx = np.vectorize(label_to_idx.get)(Z)
y_idx = np.array([label_to_idx[v] for v in y])

bg_cmap = ListedColormap(['#FFCCCC', '#FFFFAA', '#CCE5FF'])   # light pastel backgrounds
pt_cmap = ListedColormap(['#CC0000', '#CCAA00', '#0055CC'])   # bold point colours

# ── Plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# decision regions
ax.contourf(xx, yy, Z_idx, alpha=0.35, cmap=bg_cmap)
ax.contour(xx, yy, Z_idx, colors='grey', linewidths=0.5, alpha=0.5)

# training points
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                     c=y_idx, cmap=pt_cmap,
                     edgecolors='black', s=120, linewidths=1.2, zorder=5)

# legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=pt_cmap(i), markersize=10,
                       markeredgecolor='black')
           for i in range(3)]
ax.legend(handles, ['Apple', 'Banana', 'Orange'],
          title='Class', loc='upper left', fontsize=11)

ax.set_xlabel(feature_names[feat_x], fontsize=13)
ax.set_ylabel(feature_names[feat_y], fontsize=13)
ax.set_title(f'KNN Decision Boundaries  (k={k})', fontsize=15)
plt.tight_layout()
plt.show()
