from abc import ABC
from torch import nn

class BaseModel(ABC, nn.Modules):
    @abstractmethod
    def __init__(self, ):
        pass