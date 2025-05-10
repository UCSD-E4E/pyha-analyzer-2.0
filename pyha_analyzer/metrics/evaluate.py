"""
Contains Evaluation Suites for Model Training

Notes on Development:
- HF has an evaluation library, but I feel like its not well intergrated into transformers
    - See https://huggingface.co/docs/evaluate/evaluation_suite and https://huggingface.co/docs/evaluate/transformers_integrations
- For this work, I think we could create our own compute metrics group class that can handle this work
- That being said, I do like the evaluation library. I will consider allowing thier api to make this work
"""

from abc import ABC, abstractmethod
import torch

"""
Class to implement for Metrics, defines how metrics take in data and process it
"""


class Metric(ABC):
    """
    When implementing, perhaps you want to change the adveraging scheme or weight of classes, etc
    """

    def __init__(self):
        pass

    """
    Get the result of the metric
    Logits are raw outputs from model
    Labels are some target is some objective to compare against
    """

    @abstractmethod
    def __call__(self, logits=[], target=[]) -> float:
        pass


class ComputeMetricsBase(ABC):
    """
    Metrics: dict of metric names and function to slove it
    """

    def __init__(self, metrics: dict[str, Metric]):
        self.metrics_to_run = metrics

    """
    Replaces compute_metrics in hugging face 

    For each metric defined in this class, send the result
    """

    def __call__(self, eval_pred) -> dict[str, float]:
        logits = torch.Tensor(eval_pred.predictions)
        # [-1] as eval_pred.label_ids are just the model inputs...
        # print(eval_pred.label_ids, type(eval_pred.label_ids), len(eval_pred.label_ids))
        target = torch.Tensor(eval_pred.label_ids[-1]).to(torch.long)

        result = {}
        for metric_name in self.metrics_to_run.keys():
            result[metric_name] = self.metrics_to_run[metric_name](
                logits=logits, target=target
            )

        return result
