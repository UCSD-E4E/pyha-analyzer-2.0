from preprocessors import PreProcessorBase
import pandas as pd
from chunkingMethods import chunkingMethod


class OfflinePreprocessor(PreProcessorBase):
    def __init__(
                self, 
                chunk_length_s,
                min_length_s,
                overlap,
                chunk_margin_s,
                only_slide
    ):
        
        self.chunk_length_s = chunk_length_s
        self.min_length_s = min_length_s
        self.overlap = overlap
        self.chunk_margin_s = chunk_margin_s
        self.only_slide = only_slide
    



    def __call__(self, batch):
        """
            The dataset has a field detected_events (which is a list of 2-tuples) 
            and a field for events_desc (which is a list of strings mapping to each tuple)
        """
        
        old_detected_events = batch["detected_events"]
        old_events_desc = batch["events_desc"]
        clip_length = batch["end_time"]

        for item_idx in range(len(batch["audio"])):
            
            data = chunkingMethod.chunkingMethod(
                old_detected_events[item_idx],
                old_events_desc[item_idx],
                clip_length,
                chunk_length_s=self.chunk_length_s,
                min_length_s=self.min_length_s,
                overlap=self.overlap,
                chunk_margin_s=self.chunk_margin_s,
                only_slide=self.only_slide
            )   

        return batch


    
    

