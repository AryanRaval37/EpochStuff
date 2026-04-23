from Node import *
import numpy as np

# type = 1 meaning variables they are variables.
c = Node(np.array([1, 2, 3]), type=1)
e = Node(np.array([4, 5, 6]), type=1)
d = Node(np.array([7, 8, 9]), type=1)

# The way to think about the gradients at each node is:
# Change the value of the node by 1 and recompute the graph.
# the value of the parent node should be changed by the gradient value at the child node...

a = c + d
b = d * a
P = a - b
P.backward()
print(P)
