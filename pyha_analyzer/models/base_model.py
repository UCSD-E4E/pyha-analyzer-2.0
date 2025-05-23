from abc import ABC
from functools import wraps
from ..constants import MODEL_COLUMNS, REQUIRED_MODEL_OUTPUTS

"""
Should be applied on all forward functions,
If it doesn't exist, system will complain. 

Changes input to ensure function is passed the correct parameters

Should only be used for forward method of all models in this repo ideally. 
"""


# TODO rename this function, check_model_formatting()?
def has_required_inputs():
    def decorator(forward):
        @wraps(forward)
        def wrapper(self, **kwargs):
            # Checks for required inputs
            for system_expects in MODEL_COLUMNS:
                if system_expects not in kwargs:
                    raise NameError(
                        f"MISSING COLUMN IN DATASET! Please make sure `{system_expects}` is in the dataset"
                    )

            # checks for required output
            model_output = forward(self, **kwargs)
            if type(model_output) != dict:
                raise TypeError("Model output isn't a dict!")

            for expected_output in REQUIRED_MODEL_OUTPUTS:
                if expected_output not in model_output:
                    print(expected_output)
                    raise NameError(
                        f"MISSING OUTPUT IN MODEL OUTPUT! Please make sure `{expected_output}` is in the dictionary for the forward function of the model"
                    )

            return model_output

        return wrapper

    return decorator


# TODO Create System to Require Spefific Output
class BaseModel(ABC):
    """
    Regulator Class to ensure model is formated properly
    """

    def __init__(self, *args, **kwargs):
        assert hasattr(self.forward, "__wrapped__"), (
            "Please put `@has_required_inputs()` on the forward function of the model"
        )
