#Evaluation metrics
#Defining simple metrics used to evaluate correctness, hallucination risk,
#and consistency of the RAG pipeline outputs

"""
This file defines the evaluation metrics used across the project.
The metrics allow controlled comparisons between clean and poisoned settings 
in both FEVER and HotpotQA experiments.
"""

from typing import List, Dict, Any #to express the shape of the data (data structures)
from collections import Counter #to count repeated generations


#Computing accuracy for FEVER claims
#Returns 1.0 if the predicted label matches the gold label, 0.0 otherwise
def fever_accuracy(pred: str, gold: str) -> float:
    return 1.0 if pred == gold else 0.0


#Computing exact match for HotpotQA answers
#Comparison is case-insensitive and ignores extra whitespace
def hotpot_exact_match(pred: str, gold: str) -> float:
    return 1.0 if (pred or "").strip().lower() == (gold or "").strip().lower() else 0.0


#Estimating hallucination risk using verification signals
#Calculating hallucination risk: 
#   - looking at the NLI votes aggregated v = verification["votes"]
#   - if at least an evidence contradicts the claim (>0) -> risk of hallucination = 1.0
#   - otherwise = 0
def hallucination_proxy(verification: Dict[str, Any]) -> float:
    """
    More robust proxy:
    - high risk if there is explicit contradiction
    - low risk otherwise, since MNLI-based verification often predicts neutral
    """
    v = verification["votes"] #collecting aggregated NLI votes
    return 1.0 if verification.get("contradiction_strength", 0.0) >= 0.4 else 0.0

def hallucination_proxy(verification: Dict[str, Any]) -> float: #Defines a function that estimates hallucination risk from verification results
    """
    More robust proxy:
    - high risk if there is explicit contradiction
    - low risk otherwise, since MNLI-based verification often predicts neutral
    """ 
    v = verification["votes"] #Collecting the aggregated NLI votes from the verification step
    #Returning 1 if contradiction is strong, otherwise 0 (used during later evaluations)
    return 1.0 if verification.get("contradiction_strength", 0.0) >= 0.4 else 0.0 

#Measuring self-consistency across multiple generations
#Computes the proportion of the most frequent answer (majority rate)
#Calculating the self-consistency score:
#   - Considering all the generated answer <- the model generates more answers for the same claim
#   - Making the answers comparable with normalization
#   - Finding the most frequent answer
#   - Counting how many times it appears for the total number of answers

''' Example: ["yes", "yes", "no"] -> consistency score = 2 (most frequent)/ 3 ≈ 0.67 '''

def self_consistency_score(generations: List[str]) -> float:
    if not generations:
        return 0.0 #no generations -> no consistency
    norm = [g.strip().lower() for g in generations] #normalizing generated answers
    c = Counter(norm) #counting repeated answers
    top = c.most_common(1)[0][1] #frequency of the most common answer
    return top / len(norm) #returning consistency score
