#FEVER NLI pairs module
#Builds premise-hypothesis pairs from FEVER data for NLI-style training or evaluation

import random #Used for deterministic random sampling
from typing import Dict, List, Any, Tuple, Optional #Import type hints
from ragtrust.utils import norm_title #Normalize Wikipedia titles

#Label mapping compatible with roberta-large-mnli
#0=contradiction 1=neutral 2=entailment
LABEL2ID = {"REFUTES": 0, "NOT ENOUGH INFO": 1, "SUPPORTS": 2}

#Function: build a lookup table from (title sent_id) to sentence text
''' 
Input example corpus=[{"title":"Tokyo","sent_id":0,"text":"Tokyo is in Japan"}]
Output example {("Tokyo","0"):"Tokyo is in Japan"}
'''
def _build_sentence_lookup(corpus: List[Dict[str, str]]) -> Dict[Tuple[str, str], str]:
    lookup = {} #Initialize empty lookup dictionary
    for p in corpus: #Loop over corpus passages
        t = norm_title(p.get("title", "")) #Normalize page title
        sid = str(p.get("sent_id", "")) #Convert sentence id to string
        txt = (p.get("text", "") or "").strip() #Clean sentence text
        if t and sid and txt: #Keep only valid entries
            lookup[(t, sid)] = txt #Map (title sent_id) to sentence text
    return lookup #Return lookup table

#Function: extract gold evidence sentence keys from FEVER annotation
'''
Input example evidence=[[[None,None,"Tokyo",0]]]
Output example [("Tokyo","0")]
'''
def _extract_gold_sentence_keys(evidence_field: Any) -> List[Tuple[str, str]]:
    keys = [] #Initialize list of sentence keys
    if not evidence_field: #If no evidence
        return keys #Return empty list
    for ev_set in evidence_field: #Loop over evidence sets
        for item in ev_set: #Loop over individual evidence items
            if len(item) >= 4 and item[2] is not None: #Check FEVER format validity
                title = norm_title(str(item[2])) #Normalize title
                sid = str(item[3]) #Convert sentence id to string
                if title and sid: #Keep only valid keys
                    keys.append((title, sid)) #Add sentence key
    return keys #Return list of keys

#Function: create NLI-style premise-hypothesis pairs from FEVER examples
'''
Input example fever_examples=[claim,label,evidence], corpus=wiki sentences, max_pairs=100
Output example [{"premise":"...","hypothesis":"...","label":2}, ...]
'''
def make_fever_nli_pairs(
    fever_examples,
    corpus: List[Dict[str, str]],
    max_pairs: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed) #Deterministic random generator
    lookup = _build_sentence_lookup(corpus) #Build sentence lookup table

    #Random sentences used for NEI examples
    corpus_texts = [p["text"] for p in corpus if (p.get("text") or "").strip()]

    pairs: List[Dict[str, Any]] = [] #Initialize output list

    for ex in fever_examples: #Loop over FEVER examples
        claim = ex.claim #Extract claim text
        gold = ex.label #Extract gold label

        if gold in ("SUPPORTS", "REFUTES"): #If claim is supported or refuted
            keys = _extract_gold_sentence_keys(ex.evidence) #Get gold evidence keys
            for (t, sid) in keys: #Loop over gold evidence sentences
                prem = lookup.get((t, sid)) #Resolve sentence text
                if prem: #If sentence found
                    pairs.append({
                        "premise": prem,
                        "hypothesis": claim,
                        "label": LABEL2ID[gold],
                    }) #Add NLI pair
                    if len(pairs) >= max_pairs: #Stop if max reached
                        return pairs

        elif gold == "NOT ENOUGH INFO": #If claim has no supporting evidence
            for _ in range(1): #Generate one neutral example per claim
                prem = rng.choice(corpus_texts) if corpus_texts else "" #Pick random sentence
                if prem: #If sentence exists
                    pairs.append({
                        "premise": prem,
                        "hypothesis": claim,
                        "label": LABEL2ID[gold],
                    }) #Add neutral NLI pair
                    if len(pairs) >= max_pairs: #Stop if max reached
                        return pairs

    return pairs #Return all generated pairs
