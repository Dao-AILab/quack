__version__ = "0.2.2"

from quack.rmsnorm import rmsnorm
from quack.softmax import softmax
from quack.cross_entropy import cross_entropy

__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
]
