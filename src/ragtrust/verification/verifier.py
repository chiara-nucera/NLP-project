#Multi-source verification module: 
#given a claim and a list of passages (from the retriever) uses a NLI model to estimate if the evidences 
# entail/contradict/or are neutral with respect to the claim

from typing import List, Dict, Any #Import type hints 
from ragtrust.verification.nli import NLIModel #Import the project NLI wrapper used to score.

class MultiSourceVerifier: #Define the verifier that aggregates NLI signals over multiple evidence passages.
    """
    Multi-source verification in two modes:
    - mode="single": score each sentence independently (best for FEVER-style evidence).
    - mode="multi": concatenate top_n sentences and score once (best for HotpotQA multi-hop).
    """

    #Function:#Initialize the verifier with an NLI model and decision thresholds.
    def __init__(self, nli: NLIModel, thr_con: float, thr_ent: float, top_n_multi: int = 5): #Constructor with model, thresholds, and top-N for multi mode.
        self.nli = nli #Save the NLI scorer 
        self.thr_con = thr_con #Save the contradiction threshold (hyperparameter)
        self.thr_ent = thr_ent #Save the entailment threshold (hyperparameter)
        self.top_n_multi = top_n_multi #Save how many passages to concatenate in multi mode.

    #Function: #Route to the correct analysis mode ("single" or "multi") and return a verification summary.
    def analyze(self, query: str, passages: List[Dict[str, str]], mode: str = "single") -> Dict[str, Any]: #Public API to analyze evidence for a claim.
        '''
        Example input: query="Paris is in France", passages=[{"text":"Paris is the capital of France."}], mode="single"
        Example output: {"votes":{"entail":1,"contradict":0,"neutral":0},"conflict":False,...}
    '''
        if mode == "multi": #If multi-hop mode is requested
            return self._analyze_multi(query, passages) #score concatenated top-N once.
        return self._analyze_single(query, passages) #Otherwise score each passage separately.


    #Function: #Turn raw NLI scores into a single label: "entail", "contradict", or "neutral".
    def _vote_from_score(self, sc: Dict[str, float]) -> str: #Convert a score dict into a vote.
        #Rule: pick "contradict" or "entail" only if it is above threshold AND it is the larger score between the two.#Simple rule
        e = sc["entailment"] #Read entailment score from the NLI output.
        c = sc["contradiction"] #Read contradiction score from the NLI output.

        if c >= self.thr_con and c > e: #Contradict only if contradiction is strong AND stronger than entailment
            return "contradict" #Return contradiction label
        if e >= self.thr_ent and e > c: #Entail only if entailment is strong AND stronger than contradiction.
            return "entail" #Return entailment label
        return "neutral" #Otherwise return neutral (no clear decision)


    #Function: score each passage separately, count votes, and return normalized strengths plus details.
    def _analyze_single(self, query: str, passages: List[Dict[str, str]]) -> Dict[str, Any]: #Single mode: one NLI score per passage.
        '''
        Example input: query="A is B", passages=[{"title":"T","sent_id":"0","text":"A is B"}], mode: single
        Example output: {"votes":{"entail":1,"contradict":0,"neutral":0},"support_strength":1.0,"contradiction_strength":0.0,"details":[...]}
        '''
        #support_strenght = how much the evidences supports the claim (1.0 = all supports)
        #contradiction_strenght = how much the evidences contraddict the claim
        #details = list with result NLI of each evidence (testo, score, voto, poison flag)

        votes = {"entail": 0, "contradict": 0, "neutral": 0} #Start counters for each possible vote.
        details = [] #Keep per-passage info for debugging and reporting.

        for p in passages: #Loop over all retrieved passages.
            sc = self.nli.score(p["text"], query) #Compute NLI scores
            vote = self._vote_from_score(sc) #Convert scores into one label.
            votes[vote] += 1 #Add 1 vote to the chosen label.

            details.append({ #Save a record describing how this passage was judged.
                "mode": "single", #Mark that this record comes from single mode.
                "title": p.get("title", ""), #Keep title if present (otherwise empty).
                "sent_id": p.get("sent_id", ""), #Keep sentence id if present.
                "text": p.get("text", ""), #Keep the text that was scored.
                "nli": sc, #Store raw NLI scores for transparency.
                "vote": vote, #Store the final vote label.
                "is_poison": p.get("is_poison", "0"), #Keep poison metadata if available ("1" or similar).
            }) 

        conflict = votes["entail"] > 0 and votes["contradict"] > 0 #Conflict means some evidence supports and some contradicts.
        return { #Return the verification summary in a consistent schema.
            "votes": votes, #Vote counts across all passages.
            "conflict": conflict, #Whether there is mixed support vs contradiction.
            "support_strength": votes["entail"] / max(1, len(passages)), #Fraction of passages that entail the claim.
            "contradiction_strength": votes["contradict"] / max(1, len(passages)), #Fraction of passages that contradict the claim.
            "details": details, #Per-passage breakdown for inspection.
        } 
    
    #support_strength = entailing documents 
    #contradiction_strength = contradicting docs

    #Function: Concatenate the top-N passages, run NLI once, and return a single vote with details.#What it does
    def _analyze_multi(self, query: str, passages: List[Dict[str, str]]) -> Dict[str, Any]: #Multi mode: one NLI score on concatenated evidence.
        ''' 
        Example input: query="X happened in 1993", passages=[{"text":"...1993..."}, {"text":"...Tokyo..."}], top_n_multi=2
        Example output: {"votes":{"entail":1,"contradict":0,"neutral":0},"support_strength":1,"contradiction_strength":0,"details":[...]}
        '''
        votes = {"entail": 0, "contradict": 0, "neutral": 0} #Initialize vote counts
        details = [] #Keep details for the aggregated evidence.

        top = passages[: self.top_n_multi] #Take only the first top_n_multi passages.
        aggregated_text = " ".join([p.get("text", "") for p in top if p.get("text")]) #Concatenate their text into one big premise.

        sc = self.nli.score(aggregated_text, query) #Run NLI once on the aggregated premise vs the claim.
        vote = self._vote_from_score(sc) #Convert the aggregated score into one label.
        votes[vote] += 1 #Increment exactly one counter.

        details.append({ #Save a record for the aggregated decision.
            "mode": "multi", #Mark that this record comes from multi mode.
            "text": aggregated_text, 
            "nli": sc, #Store raw NLI scores.
            "vote": vote, #Store the chosen label.
        }) 

        conflict = votes["entail"] > 0 and votes["contradict"] > 0 #Usually False here because there is only one vote.
        return { #Return a schema consistent with single mode.
            "votes": votes, #Vote counts (0/1 in practice).
            "conflict": conflict, #Conflict flag for consistency with other mode.
            "support_strength": votes["entail"], #In multi mode this is 0 or 1 because we made one decision.
            "contradiction_strength": votes["contradict"], #In multi mode this is 0 or 1 because we made one decision.
            "details": details, 
        } 
