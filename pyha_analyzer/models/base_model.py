from abc import ABC, abstractmethod
from torch import nn
from functools import wraps
from ..constants import DEFAULT_COLUMNS

"""
Should be applied on all forward functions,
If it doesn't exist, system will complain. 

Changes input to ensure function is passed the correct parameters

Should only be used for forward method of all models in this repo ideally. 
"""
def has_required_inputs(required_input_columns):
    def decorator(forward):
        @wraps(forward)
        def wrapper(self, **kwargs):
            for system_expects in DEFAULT_COLUMNS:
                if system_expects not in kwargs:
                    return f"MISSING COLUMN IN DATASET! Please make sure {system_expects} is in the dataset"
            return forward(self, **kwargs)
        return wrapper
    return decorator

class BaseModel(ABC, nn.Module):
    @abstractmethod
    def __init__(self, ):
        pass