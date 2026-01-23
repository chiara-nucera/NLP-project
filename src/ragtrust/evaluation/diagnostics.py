"""
Additional diagnostic metrics for the RAGTrust project.
(retrieval recall@k and conflict statistics)
"""

def evidence_recall_at_k(pipeline, examples, k=10): #Defines a function to compute evidence recall at k
    """
    Measures whether the retriever finds at least one correct evidence
    among the top-k retrieved sentences.
    """
    hits = 0 #Counts how many claims have at least one correct evidence
    total = 0 #Counts total number of evaluated claims

    for ex in examples: #Loops over all examples
        retrieved = pipeline.retriever.retrieve(ex.claim, k=k) #Retrieves top-k passages for the claim
        retrieved_titles = {r["title"] for r in retrieved} #Collects titles of retrieved passages

        gold_titles = set() #Initializes set of gold evidence titles
        for ev_set in ex.evidence:
            for ev in ev_set:
                if ev[2] is not None:
                    gold_titles.add(ev[2]) #Adds gold evidence title

        if retrieved_titles & gold_titles: #Checks if any retrieved title matches a gold title
            hits += 1 #Counts a successful retrieval
        total += 1 #Increments total number of examples

    return hits / total if total > 0 else 0.0 #Returns recall@k or 0 if no examples
