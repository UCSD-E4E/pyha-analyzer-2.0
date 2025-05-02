from abc import ABC
from torch import nn

class Base_Model(ABC, nn.Modules):
    @abstractmethod
    def __init__(self, preprocessor, ):
        while False:
            yield None