from .evaluate import Metric, ComputeMetricsBase

# TODO: should the metric define the name being used?

class cMAP(Metric):
    """
    top_n looks at performance from the top_n number of species
    agg diffrent aggerations of the metric across classes
    """
    def __init__(self, top_n=-1, agg="mean"):
        pass

    def __call__(self, logits=[], target=[]) -> float:
        pass

class ROCAUC(Metric):
    def __init__(self):
        pass
    
    def __call__(self, logits=[], target=[]) -> float:
        pass


class AudioClassificationMetrics(ComputeMetricsBase):
    def __init__(self, metrics, class_size=-1): # Is class size assumed?
        if len(metrics) > 0:
            raise "WARNING, THIS DOES NOT TAKE IN EXTRA METRICS. Discuss with Project Leads before moving forward."
        
        self.metrics = {
            "cMAP": cMAP(),
            "cMAP-5": cMAP(top_n=5),
            "ROCAUC": ROCAUC()
        }

        super().__init__(metrics)