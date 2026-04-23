
# ! Parts of this file are AI generated

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from DecisionTree import DecisionTree
from impurity_metrics import gini

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

X = data[:, :-1].astype(float)
y_raw = data[:, -1]

## One-hot encoding again
label_map = {'Wine': 1, 'Beer': 2, 'Whiskey': 4}
reverse_label_map = {v: k for k, v in label_map.items()}
y = np.array([label_map[label] for label in y_raw])

# ── Pick two features to plot ─────────────────────────────────────────
# Feature 0: Alcohol,  Feature 1: Sugar  (most visually separable)
feat_x, feat_y = 0, 1
feature_names = ['Alcohol Content (%)', 'Sugar (g/L)', 'Color']

X_2d = X[:, [feat_x, feat_y]]

# ── Fit Decision Tree on the 2-feature subset ────────────────────────
dt = DecisionTree(max_depth=5, min_samples_split=2, impurity_metric=gini)
dt.fit(X_2d, y)

# ── Build decision boundary mesh ─────────────────────────────────────
h = 0.2  # step size
x_min, x_max = X_2d[:, 0].min() - 2, X_2d[:, 0].max() + 2
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))

mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = np.array(dt.predict(mesh_points))
Z = Z.reshape(xx.shape)

# ── Colours ───────────────────────────────────────────────────────────
# Map encoded labels (1, 2, 4) → indices (0, 1, 2) for colourmaps
label_to_idx = {1: 0, 2: 1, 4: 2}
Z_idx = np.vectorize(label_to_idx.get)(Z)
y_idx = np.array([label_to_idx[v] for v in y])

bg_cmap = ListedColormap(['#E8D0FF', '#FFFFAA', '#CCE5FF'])   # light pastel backgrounds
pt_cmap = ListedColormap(['#7700AA', '#CCAA00', '#0055CC'])   # bold point colours

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
ax.legend(handles, ['Wine', 'Beer', 'Whiskey'],
          title='Class', loc='upper right', fontsize=11)

ax.set_xlabel(feature_names[feat_x], fontsize=13)
ax.set_ylabel(feature_names[feat_y], fontsize=13)
ax.set_title('Decision Tree - Decision Boundaries', fontsize=15)
plt.tight_layout()
plt.show()
