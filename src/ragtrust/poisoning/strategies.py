#Poisoning strategies module:
# Applies synthetic poisoning to retrieved passages or to the full corpus.

from typing import List, Dict #Import type hints
from ragtrust.data.synthetic import inject_synthetic_poison #Function to inject poisoned passages

class Poisoner:
    #Function: initialize the poisoner
    #Store poisoning strategy, rate, and random seed
    def __init__(self, strategy: str, rate: float, seed: int):
        '''Input example: strategy="contradictory_passage", rate=0.1, seed=42
        Output example: A Poisoner object storing strategy, rate, and seed.
        '''
        self.strategy = strategy #Name of the poisoning strategy
        self.rate = rate #Fraction of data to poison
        self.seed = seed #Random seed for reproducibility

    #Function: apply poisoning to retrieved passages
    def apply(self, retrieved: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if self.rate <= 0: #If poisoning is disabled
            return retrieved #Return original retrieved passages
        if self.strategy == "contradictory_passage": #If using contradictory passage strategy
            return inject_synthetic_poison(
                retrieved,
                rate=self.rate,
                seed=self.seed
            ) #Inject synthetic contradictory passages
        return retrieved #Fallback: return original passages

    #Function: apply poisoning directly to the full corpus
    #-> create a poisoned version of the dataset before retrieval
    def apply_to_corpus(self, corpus: List[Dict[str, str]]) -> List[Dict[str, str]]:
        '''
        Input example: retrieved=[{"text":"Paris is in France","is_poison":"0"}, {...}]
        Output example: same list with some passages modified and marked as poisoned.
        '''
        if self.rate <= 0: #If poisoning is disabled
            return corpus #Return original corpus

        import random #Used for random sampling
        from ragtrust.data.synthetic import make_contradictory_passage #Create contradictory text

        rng = random.Random(self.seed) #Create deterministic random generator
        out = [dict(p) for p in corpus] #Copy corpus entries

        n_poison = int(len(out) * self.rate) #Compute number of passages to poison
        idxs = rng.sample(
            range(len(out)),
            k=max(0, min(n_poison, len(out)))
        ) #Select random indices to poison

        for i in idxs: #Loop over selected passages
            out[i]["text"] = make_contradictory_passage(out[i].get("text", "")) #Replace text with contradictory version
            out[i]["is_poison"] = "1" #Mark passage as poisoned
            out[i]["doc_id"] = out[i].get("doc_id", "") + "__POISON" #Tag document id

        return out #Return poisoned corpus
