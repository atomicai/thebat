import abc
from typing import Union, List

class IR(abc.ABC):
    
    @abc.abstractmethod
    def retrieve_topk(self, query: Union[str, List[str]], **kwargs):
        pass


__all__ = ["IR"]