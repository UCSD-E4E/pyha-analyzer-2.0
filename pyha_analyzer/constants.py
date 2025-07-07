DEFAULT_COLUMNS = [
    "audio",  # Considered to be original audio
    "audio_in",  # Considered to be model output
    "labels",  # Considered to be label for loss
    "filepath", # There should always be a file this data point is associated with
]

MODEL_COLUMNS = [
    "audio",  # Considered to be original audio
    "audio_in",  # Considered to be model output
    "labels",  # Considered to be label for loss
]

REQUIRED_MODEL_OUTPUTS = ["logits", "loss"]

DEFAULT_PROJECT_NAME = "pa2.0_test" # TODO: change?
DEFAULT_RUN_NAME = "working_dir" # TODO: change? 