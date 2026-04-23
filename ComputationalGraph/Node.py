import numpy as np  

# Future upgrades:
# - consider heavy optimization for speed - some jit and some shortcuts for operations
# - more general design for operators

# here its a node class but actually the root Node itself will be the whole graph
# root node must have all the children
class Node:
    def __init__(self, value, child1=None, child2=None, op=None, op_args=None, type=1):
        # Value is an vector
        self.value = value
        self.child1 = child1
        self.child2 = child2

        # 1 if variable
        # 0 if constant
        self.type = type

        # gradients to be accumulated during backpropagation, initialized to 0
        self.grad = 0
        self.grad_calculated = False
        self.requires_grad = True if self.type == 1 else False

        # design choice: should op be a node or just be stored in the parent node of the having the multiple children nodes?
        # NOTE: keep op wtih parent node - op is part of the link of the nodes, not the node itself.
        self.op = op

        # Another design choice:
        # - have separate nodes for variables and consts or should the consts be part of the parent node itself stored in op args
        # - currently its different node for consts - op_args is not used.
        self.op_args = op_args


    # figure out better way to pass up what function it is instead of a string saying 'add'
    def __add__(self, other):
        # add operation
        # create a new node with value as the sum of the two nodes, and children as the two nodes
        # here op is add (children are added to get to this node)
        # type should be variable if either of self or other is variable
        return Node(self.value + other.value, child1=self, child2=other, op='add', type = max(self.type, other.type))
    
    def __mul__(self, other):
        # mul operation
        return Node(self.value * other.value, child1=self, child2=other, op='mul', type = max(self.type, other.type))
    
    def __sub__(self, other):
        # sub operation
        return Node(self.value - other.value, child1=self, child2=other, op='sub', type = max(self.type, other.type))   
    
    # division
    def __truediv__(self, other):
        return Node(self.value / other.value, child1=self, child2=other, op='div', type = max(self.type, other.type))
    
    # exponent
    def __pow__(self, other):
        return Node(self.value ** other.value, child1=self, child2=other, op='pow', type = max(self.type, other.type))
    
    # thought of adding forward pass function but as you add things to the node, everything is already computed.
    # design choice: is it better to have precomputed values or have a forward pass function
    # if forward pass what should be done when you add stuff to the graph? 
    # probably just store make the tree - just store the children without actually doing anything

    # GRADIENT CALCULATION:

    # recursive function to calculate gradients.
    # NOTE: the double gradient calculation problem
    # happens when more than one output from node. - have requires grad flag
    def backward(self, incoming_grad=None):

        # this is to fix the double gradient problem 
        # if you have already calculated the gradient for a node, you should not calculate it again because it will mess up the gradients
        # pass the sum of the gradients from the different paths to the node and then calculate the gradient for that node only once.
        if incoming_grad is None:
            # this is resetting the gradients for all the children node to 0
            # only when the last incoming grad comes in, we still reset to 0 and then proceed the function which will set the gradients correctly wrt new self.grad accumulated values.
            visited = set()
            stack = [self]
            while stack:
                node = stack.pop()
                node_id = id(node)
                if node_id in visited:
                    continue
                visited.add(node_id)
                node.grad = np.zeros_like(node.value, dtype=np.float64)
                node.grad_calculated = False
                if node.child1 is not None:
                    stack.append(node.child1)
                if node.child2 is not None:
                    stack.append(node.child2)
            incoming_grad = np.ones_like(self.value, dtype=np.float64)
        else:
            incoming_grad = np.asarray(incoming_grad, dtype=np.float64)

        self.grad_calculated = True
        if self.requires_grad:
            self.grad += incoming_grad

        ## GOAL OF FUNCTION:
        #  task is to find the derivative of the parent node with respect to the current node.
        #  parent node - whatever that is - should it be mentioned or do i just keep going up the tree?
        #  no need to "find parent" just go down from which ever node this function is called on.
        #  every node should have the gradient of how the parent node would change if this node changes.

        # for example parent node P
        # P = A + B
        # dP/dA = 1 if this is the only connection going up from A then gradient of A is just 1
        # dP/dB = 1
        # now if A = C * D
        # dP/dC = dP/dA * dA/dC = 1 * D = D
        # dP/dD = dP/dA * dA/dD = 1 * C = C
        # now if B = C * E
        # dP/dC is already D from the previous calculation but now we have another path to C through B
        # we just accumulate this gradient - dP/dC = D + dP/dB * dB/dC = D + 1 * E = D + E
        # dP/dE = dP/dB * dB/dE = 1 * C = C


        # Base case: see if you're at the leaf node (variable or constant)
        if self.child1 is None and self.child2 is None:
            return

        # Recursive case: compute gradients for children based on the operation
        child1_grad = None
        child2_grad = None

        if self.op == 'add':
            # gradient of addition is just 1 for both children
            if self.child1.requires_grad:
                child1_grad = incoming_grad  # dP/dchild1 = dP/dself * dself/dchild1 = grad * 1
            if self.child2.requires_grad:
                child2_grad = incoming_grad  # dP/dchild2 = dP/dself * dself/dchild2 = grad * 1
            
        elif self.op == 'mul':
            # gradient of multiplication is the other child
            if self.child1.requires_grad:
                child1_grad = incoming_grad * self.child2.value  # dP/dchild1 = dP/dself * dself/dchild1 = grad * child2
            if self.child2.requires_grad:
                child2_grad = incoming_grad * self.child1.value  # dP/dchild2 = dP/dself * dself/dchild2 = grad * child1
        
        elif self.op == 'sub':
            # gradient of subtraction is 1 for child1 and -1 for child2
            if self.child1.requires_grad:
                child1_grad = incoming_grad  # dP/dchild1 = dP/dself * dself/dchild1 = grad * 1
            if self.child2.requires_grad:
                child2_grad = -incoming_grad  # dP/dchild2 = dP/dself * dself/dchild2 = grad * -1

        elif self.op == 'div':
            # gradient of division is 1/child2 for child1 and -child1/(child2^2) for child2
            if self.child1.requires_grad:
                child1_grad = incoming_grad / self.child2.value  # dP/dchild1 = dP/dself * dself/dchild1 = grad * (1/child2)
            if self.child2.requires_grad:
                child2_grad = -incoming_grad * self.child1.value / (self.child2.value ** 2)  # dP/dchild2 = dP/dself * dself/dchild2 = grad * (-child1/(child2^2))

        elif self.op == 'pow':
            # gradient of power is child2 * (child1^(child2-1)) for child1 and log(child1) * (child1^child2) for child2
            if self.child1.requires_grad:
                child1_grad = incoming_grad * self.child2.value * (self.child1.value ** (self.child2.value - 1))  # dP/dchild1 = dP/dself * dself/dchild1 = grad * (child2 * (child1^(child2-1)))
            if self.child2.requires_grad:
                child2_grad = incoming_grad * np.log(self.child1.value) * (self.child1.value ** self.child2.value)  # dP/dchild2 = dP/dself * dself/dchild2 = grad * (log(child1) * (child1^child2))

        # now that gradients for children are calculated, we can call backward on them to propagate the gradients down the tree
        if self.child1 is not None and child1_grad is not None:
            self.child1.backward(child1_grad)
        if self.child2 is not None and child2_grad is not None:
            self.child2.backward(child2_grad)
    

    # ! AI generated function to plot graphs
    def __repr__(self):
        from graphviz import Digraph
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from io import BytesIO

        dot = Digraph(format='png')
        dot.attr(rankdir='BT', dpi='300')
        dot.attr('node', shape='box', style='filled', fontsize='11')

        visited = set()

        def _add_nodes(node):
            node_id = str(id(node))
            if node_id in visited:
                return
            visited.add(node_id)

            type_str = 'Var' if node.type == 1 else 'Const'
            title = type_str
            if node.op:
                title = node.op

            value_str = np.array2string(node.value)
            if node.grad_calculated:
                grad_str = np.array2string(node.grad)
                label = (
                    f'<<FONT POINT-SIZE="12">{title}</FONT><BR/>'
                    f'<FONT POINT-SIZE="10">value = {value_str}</FONT><BR/><BR/>'
                    f'<FONT POINT-SIZE="10">grad = {grad_str}</FONT>>'
                )
            else:
                label = (
                    f'<<FONT POINT-SIZE="12">{title}</FONT><BR/>'
                    f'<FONT POINT-SIZE="10">value = {value_str}</FONT>>'
                )

            color = '#FFD580' if node.op else ('#ADD8E6' if node.type == 1 else '#D3D3D3')
            dot.node(node_id, label=label, fillcolor=color)

            for child in [node.child1, node.child2]:
                if child is not None:
                    child_id = str(id(child))
                    _add_nodes(child)
                    dot.edge(child_id, node_id)

        _add_nodes(self)

        # render high-res PNG in memory, no file saved
        png_bytes = dot.pipe()
        buf = BytesIO(png_bytes)
        img = mpimg.imread(buf)

        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Computation Graph')
        plt.tight_layout()
        plt.show()

        return f"Node(value={self.value}, type={'Variable' if self.type == 1 else 'Constant'}, op={self.op})"
