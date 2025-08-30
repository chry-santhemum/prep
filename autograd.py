
from typing import Optional

class Scalar:
    def __init__(self, data: Optional[float]):
        self.data: float = 0. if data is None else data
        self.grad: float|None = 0.  # None means don't capture grad
        self.grad_this_pass: float=0.
        self._prev: set["Scalar"] = {}

    def __add__(self, other: "int|float|Scalar"):
        if isinstance(other, Scalar):
            out = Scalar(data=self.data + other.data)
            out._prev = {self, other}
        
        else:
            other_scalar = Scalar(data=float(other))
            other_scalar.grad = None  # Don't track gradients
            out = Scalar(data=self.data + other_scalar.data)
            out._prev = {self, other_scalar}

        def add_backward():
            if self.grad is not None:
                self.grad += out.grad_this_pass
            self.grad_this_pass += out.grad_this_pass
            if other.grad is not None:
                other.grad += out.grad_this_pass
            other.grad_this_pass += out.grad_this_pass
        
        out._backward = add_backward

        return out

    def __mul__(self, other: "int|float|Scalar"):
        if isinstance(other, Scalar):
            out = Scalar(data=self.data * other.data)
            out._prev = {self, other}
        
        else:
            other_scalar = Scalar(data=float(other))
            other_scalar.grad = None  # Don't track gradients
            out = Scalar(data=self.data * other_scalar.data)
            out._prev = {self, other_scalar}

        def mul_backward():
            if self.grad is not None:
                self.grad += out.grad_this_pass * other.data
            self.grad_this_pass += out.grad_this_pass * other.data
            if other.grad is not None:
                other.grad += out.grad_this_pass * self.data
            other.grad_this_pass += out.grad_this_pass * self.data
        
        out._backward = mul_backward

        return out

    def __neg__(self):
        out = Scalar(data=-self.data)
        out._prev = {self}
        def neg_backward():
            if self.grad is not None:
                self.grad -= out.grad_this_pass
            self.grad_this_pass -= out.grad_this_pass
        
        out._backward = neg_backward
        return out

    def __sub__(self, other):
        neg_other = other.__neg__()
        out = self.__add__(neg_other)
        return out


    def _backward(self):
        pass

    @staticmethod
    def _visit(node: "Scalar", marked: "list[Scalar]", traversed: "list[Scalar]"):
        # marked: nodes that are in the current DFS path
        # modifies marked and traversed in-place
        # this operation conserves len(marked)
        if node in marked:
            raise ValueError("Computation graph has a directed cycle.")
        marked.append(node)
        for parent in node._prev:
            Scalar._visit(parent, marked, traversed)
        marked.pop(-1)
        traversed.insert(0, node)
        print(f"current traversed: {[node.data for node in traversed]}")


    def dfs(self) -> "list[Scalar]":
        """
        Returns the list of self and its _prevs,
        such that a node comes before all its ancestors.
        """
        marked, traversed = [], []
        Scalar._visit(self, marked, traversed)
        return traversed


    def backward(self):
        traversed = self.dfs()
        for node in traversed:
            node.grad_this_pass = 0.
        self.grad_this_pass = 1.

        for node in traversed:
            node._backward()
