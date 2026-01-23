#Hybrid retrieval module: Combines BM25 and dense retrieval results into a single ranked list

from typing import List, Dict, Tuple #Import type hints
from ragtrust.indexing.bm25 import BM25Retriever #Import BM25 retriever
from ragtrust.indexing.dense import DenseRetriever #Import dense retriever

class HybridRetriever:
    #Function: initialize a hybrid retriever -> optionally enable BM25, dense retrieval, or both
    def __init__(
        self,
        corpus: List[Dict[str, str]],
        use_bm25: bool,
        use_dense: bool,
        dense_model: str,
        bm25_tokenizer: str
    ):
        self.corpus = corpus #Store the corpus
        self.use_bm25 = use_bm25 #Flag to enable BM25
        self.use_dense = use_dense #Flag to enable dense retrieval
        self.bm25 = BM25Retriever(corpus, bm25_tokenizer) if use_bm25 else None #Create BM25 retriever if enabled
        self.dense = DenseRetriever(corpus, dense_model) if use_dense else None #Create dense retriever if enabled

    #Function: retrieve passages using BM25, dense, or both, and merge results.
    def retrieve(self, query: str, k: int) -> List[Dict[str, str]]:
        scored: List[Tuple[float, Dict[str, str]]] = [] #Collect (score, passage) pairs
        #score is a retriever score to indicate if a sentence is relevant for the query

        if self.bm25: #If BM25 is enabled
            scored.extend(self.bm25.retrieve(query, k)) #Add BM25 results
        if self.dense: #If dense retrieval is enabled
            scored.extend(self.dense.retrieve(query, k)) #Add dense results

        #Merge duplicates using (title, sent_id, text) as key and keep max score
        best = {} #Map passage key to best (score, passage)
        for s, p in scored: #Loop over all retrieved results
            key = (p.get("title",""), p.get("sent_id",""), p.get("text","")) #Define unique passage key
            if key not in best or s > best[key][0]: #Keep only the highest score for this passage
                best[key] = (s, p)

        merged = sorted(best.values(), key=lambda x: x[0], reverse=True)[:k] #Sort by score and keep top-k
        out = [] #Final output list
        for s, p in merged: #Loop over merged results
            q = dict(p) #Copy passage dictionary
            q["score"] = str(s) #Attach retrieval score as string
            out.append(q) #Add to output list
        return out #Return final ranked passages
