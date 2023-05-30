from abc import ABC
from numpy import array


class BaseImageOperation(ABC):
    """
    Base class for all image operations. Provides an interface for successor classes. 
    """
    
    _op = None

    def process(self, X: array) -> array:
        if not self._op:
            raise NotImplemented
        
        return self._op(X)

    def __call__(self, X: array, *args, **kwargs) -> array:
        if not self._op:
            raise NotImplemented
        
        return self._op(X)
