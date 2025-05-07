from abc import ABC, abstractmethod
from ..trainer import PyhaTrainer

"""
Create Shared Logging System for Common Metrics
"""


class Logger(ABC):
    def __init__(self):
        pass

    """
    How does the trainer get configured by the logger?
    Return the updated trainer, all ready to go for logging.
    """
    @abstractmethod
    def modify_trainer(self, trainer) -> PyhaTrainer:
        pass


    """
    Less descturctive method than `modify_trainer`

    Basically if Transformer library supports logging method already, just change training arg
    """
    @abstractmethod
    def modify_training_args(working_dir, training_args=None):
        pass

    """
    
    """

