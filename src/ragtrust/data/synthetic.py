#Data preparation
#Creating synthetic contradictory passages to simulate poisoned evidence in a controlled way

"""
This file defines simple utilities to generate synthetic poisoned passages by introducing
naive contradictions into retrieved texts. The goal is to simulate unreliable or adversarial
evidence in a controlled setting, allowing the evaluation of robustness and trustworthiness
in the RAG pipeline.
"""

from typing import List, Dict #to annotate expected data structures
import random #to ensure reproducible random poisoning


#Creating a naive contradictory version of a passage
#Input: string <- sentence from Wikipedia
#If not is already contained in the sentence, it is removed, otherwise a negation is added
#=> Output: contraddicting version of the original text
def make_contradictory_passage(text: str) -> str:
    """
    Baseline to generate a contradiction.
    It is not linguistically perfect, but sufficient for controlled experiments.
    """
    t = text.strip()
    if not t:
        return "This statement is false."
    #Heuristic: remove explicit negation if present
    if " not " in t.lower():
        return t.replace(" not ", " ")
    #Fallback: prefix a generic negation
    return "It is not true that: " + t


#Injecting synthetic poison by modifying a subset of retrieved passages in place
#Input: 
#       retrieved = list of dictionaries, each is a passage of the retriever {"doc_id", "title", "sent_id", "text", }
#       rate = % indicating how many passages to poison
def inject_synthetic_poison(retrieved, rate, seed=0):
    rng = random.Random(seed) #initializing deterministic random generator
    
    #Each passage is added to out
    out = [dict(p, is_poison=False) for p in retrieved]
    
    n_poison = max(1, int(len(out) * rate))  #calculating the number of passages to poison 
    
    #The indexes are used to select the casual sentences that will be poisoned
    idxs = rng.sample(range(len(out)), k=max(0, min(n_poison, len(out)))) #selecting random indices
    #range(len(out)) -> all the possible indexes
    #min(n_poison, len(out) -> avoiding to ask more indexes than the existing number
    #max -> avoiging negative values if rate is really small

    for i in idxs:
        #substituting the original text with the contraddicting version
        out[i]["text"] = make_contradictory_passage(out[i]["text"]) 

        #Signaling the poison
        out[i]["is_poison"] = True
        out[i]["source"] = "poison"
        out[i]["doc_id"] = out[i]["doc_id"] + "__POISON" #updating document identifier

    return out 

#returning the poisoned corpus <- mix clean + poisoned corpus but w/o duplications
