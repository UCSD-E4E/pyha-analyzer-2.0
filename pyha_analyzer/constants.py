DEFAULT_COLUMNS = [
    "audio",  # Considered to be original audio
    "audio_in",  # Considered to be model output
    "labels",  # Considered to be label for loss
    "filepath", # There should always be a file this data point is associated with
    "detected_events", # For handling strong labels
]

MODEL_COLUMNS = [
    "audio",  # Considered to be original audio
    "audio_in",  # Considered to be model output
    "labels",  # Considered to be label for loss
    "detected_events",
]

REQUIRED_MODEL_OUTPUTS = ["logits", "loss"]
