from abc import ABC, abstractmethod

"""
Base class for preprocessing

Could be online or offline, but always follows this class
"""


class PreProcessorBase(ABC):
    def __init__(self, 
             name:str,
             *args,
             **kwargs):
        self.name = name

    @abstractmethod
    def __call__(self, batch):
        pass
