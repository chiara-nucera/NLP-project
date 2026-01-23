#Evaluation utilities
#Aggregating evaluation metrics over batches of FEVER and HotpotQA examples

"""
This file defines helper functions to aggregate (calculating the sample) evaluation metrics over a batch
of examples. It does not compute metrics itself. The same aggregation logic is reused for both
FEVER and HotpotQA settings.
"""
#It receives a list of examples evaluated line by line + their metrics accuracy, hallucination, self-consistency
#and for each metric calculates the average on the batch

from typing import List, Dict, Any, Optional #to annotate expected data structures
from ragtrust.evaluation.metrics import (
    fever_accuracy, hotpot_exact_match, hallucination_proxy, self_consistency_score
) #importing metric functions in the evaluation pipeline


#Aggregating per-example evaluation results into average metrics
def aggregate(results: List[Dict[str, Any]]) -> Dict[str, float]:
    '''Example of input:
    results = [
    {"acc": 1.0, "hallucination": 0.0, "self_consistency": 0.8},
    {"acc": 0.0, "hallucination": 1.0, "self_consistency": 0.6},
    {"acc": 1.0, "hallucination": 0.0, "self_consistency": 0.9},
]
    where each dictionary is an example (evaluated) 

    Example of output:
    {"acc": 0.67, "hallucination": 0.33, "self_consistency": 0.77}
    For a batch of examples
'''
    if not results:
        return {} #return empty dict if there are no results

    keys = ["acc", "hallucination", "self_consistency"] #metrics to be aggregated
    out = {}
    for k in keys:
        #collecting metric values if present
        vals = [r.get(k) for r in results if r.get(k) is not None]
        if vals:
            out[k] = sum(vals) / len(vals) #computing the average value
    return out


#Aggregating evaluation results for a batch of FEVER examples
def eval_fever_batch(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return aggregate(rows) #using the same aggregation logic


#Aggregating evaluation results for a batch of HotpotQA examples
def eval_hotpot_batch(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return aggregate(rows) #using the same aggregation logic
