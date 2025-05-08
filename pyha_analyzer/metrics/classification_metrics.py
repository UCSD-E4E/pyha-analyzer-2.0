import numpy as np
from torchmetrics.classification import (
    MultilabelAveragePrecision, MulticlassAveragePrecision,
    MulticlassAUROC, MultilabelAUROC,

)
from .evaluate import Metric, ComputeMetricsBase


# TODO: should the metric define the name being used?
class cMAP(Metric):
    """ 
    Mean average precision metric for a batch of outputs and labels 
    Returns tuple of (class-wise mAP, sample-wise mAP)
    """
    
    def __init__(self, num_classes, mutlilabel=True, top_n=-1, samplewise=False):
        """
        top_n looks at performance from the top_n number of species
        agg diffrent aggerations of the metric across classes
        """
       
        if mutlilabel:
            self.metric = MultilabelAveragePrecision(num_labels=num_classes, average="none")
        else:
            self.metric = MulticlassAveragePrecision(num_classes=num_classes, average="none")

        self.num_classes = num_classes

    def __call__(self, logits=[], target=[]) -> float:
        map_by_class = self.metric(logits, target)
        cmap = map_by_class.nanmean()
        #TODO FIX This to define weight based on number of targets per class in batch
        #smap = (map_by_class * class_dist/class_dist.sum()).nansum() 

        # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
        # Could be possible when model is untrained so we only have FNs
        if np.isnan(cmap):
            return 0
        return cmap.item()
    
class ROCAUC(Metric):
    def __init__(self, num_classes, mutlilabel=True):
        if mutlilabel:
            self.metric = MultilabelAUROC(num_labels=num_classes, average="none")
        else:
            self.metric = MulticlassAUROC(num_classes=num_classes, average="macro")

        self.num_classes = num_classes

    def __call__(self, logits=[], target=[]) -> float:
        map_by_class = self.metric(logits, target)
        auroc = map_by_class.nanmean()

        # https://forums.fast.ai/t/nan-values-when-using-precision-in-multi-classification/59767/2
        # Could be possible when model is untrained so we only have FNs
        if np.isnan(auroc):
            return 0
        return auroc.item()


class AudioClassificationMetrics(ComputeMetricsBase):
    def __init__(self, metrics, num_classes=-1, mutlilabel=True): # Is class size assumed?
        if len(metrics) > 0:
            raise "WARNING, THIS DOES NOT TAKE IN EXTRA METRICS. Discuss with Project Leads before moving forward."
        
        self.metrics = {
            "cMAP": cMAP(num_classes, mutlilabel=False), #TODO handle mutlilabel better
            #"cMAP-5": cMAP(num_classes, mutlilabel=mutlilabel, top_n=5),
            "ROCAUC": ROCAUC(num_classes, mutlilabel=False)
        }

        super().__init__(self.metrics)