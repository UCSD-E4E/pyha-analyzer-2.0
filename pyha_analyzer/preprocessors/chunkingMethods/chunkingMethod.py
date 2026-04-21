
from typing import Dict, List, Tuple

'''
These methods are adapted from pyha-analyzer-1.0!
'''

def convolving_chunk(detected_events: List[Tuple[int, int]],
                     events_desc: List[str],
                     clip_length: float,
                     chunk_length_s: int, 
                     min_length_s: float, 
                     overlap: float,
                     chunk_margin_s: float, 
                     only_slide: bool) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Helper function that converts a binary annotation row to uniform chunks. 
    Note: Annotations of length shorter than min_length are ignored. Annotations
    that are shorter than or equal to chunk_length are chopped into three chunks
    where the annotation is placed at the start, middle, and end. Annotations
    that are longer than chunk_length are chunked used a sliding window.
    Args:
        detected_events (List[Tuple[int, int]])
            - List of 2-tuples where an event occurs
        events_desc (List[str])
            - List of strings corresponding to said events
        clip_length_s
            - Length of the clip
        chunk_length_s (int)
            - Duration in seconds to set all annotation chunks
        min_length_s (float)
            - Duration in seconds to ignore annotations shorter in length
        overlap (float)
            - Percentage of overlap between chunks
        chunk_margin_s (float)
            - Duration to pad chunks on either side
        only_slide (bool)
            - If True, only annotations greater than chunk_length_s are chunked
    Returns:
        Chunked detected_events and events_desc lists
    """

    new_detected_events = []
    new_events_desc = []
    for index in range(len(detected_events)):
          
          time_pair = detected_events[index]
          offset_s = max(float(time_pair[0]-chunk_margin_s), 0) # if this becomes negative, take 0
          duration_s = float(time_pair[1] - time_pair[0])
          duration_s += 2 * chunk_margin_s # on both sides
          end_s = min(offset_s + duration_s, clip_length)
          chunk_self_time = chunk_length_s * (1 - overlap)


          #Ignore small duration (could be errors, play with this value)
          if duration_s < min_length_s:
                return ([], [])
          
          #calculate valid offsets for short annotations

          if duration_s <= chunk_length_s:
                # start of clip
                if (offset_s + chunk_length_s) < float(clip_length) and not only_slide:
                    new_detected_events.append((offset_s, offset_s + chunk_length_s))
                    new_events_desc.append(events_desc[index])
                # middle of clip
                if (end_s - chunk_length_s/2.0) > 0 and (end_s + chunk_length_s/2.0) < clip_length:
                      new_detected_events.append(((offset_s + end_s)/2.0 - chunk_length_s/2.0, (offset_s + end_s)/2.0))
                      new_events_desc.append(events_desc[index])
                # end of clip
                if (end_s - chunk_length_s) > 0 and not only_slide:
                      new_detected_events.append((end_s - chunk_length_s, end_s))
                      new_events_desc.append(events_desc[index])

                # calculate valid offsets for long annotations

          else: 
                # how many clips there would be
                clip_num = int(round(duration_s / chunk_self_time))
                for i in range(clip_num):
                    if (offset_s + chunk_length_s) + (i * chunk_self_time) <= clip_length:
                            new_detected_events.append((offset_s + i * chunk_self_time, offset_s + i * chunk_self_time + chunk_length_s))
                            new_events_desc.append(events_desc[index])
    
    return new_detected_events, new_events_desc


def chunkingMethod(
            detected_events: List[Tuple[int, int]],
            events_desc: List[str],
            clip_length: float, 
            chunk_length_s: int = 5,
            min_length_s: int = 0,
            overlap: float = 0.5,
            chunk_margin_s: float = 0,
            only_slide: bool = False
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        new_detected_events, new_events_desc = convolving_chunk(
            detected_events,
            events_desc, 
            clip_length,
            chunk_length_s,
            min_length_s,
            overlap,
            chunk_margin_s,
            only_slide
        )

        return new_detected_events, new_events_desc

