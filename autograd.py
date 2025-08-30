
from typing import Optional

class Scalar:
    def __init__(self, data: Optional[float]):
        self.data: float = 0. if data is None else data
        self.grad: float = 0
        self._prev: Optional[set["Scalar"]] = None

    def __add__(self, other):
        