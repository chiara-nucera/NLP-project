#BM25 retrieval module:#Implements a simple BM25-based retriever over a text corpus.

from typing import List, Dict, Tuple #Import type hints 
from rank_bm25 import BM25Okapi #Import BM25 implementation.

#Function: tokenize text into a list of terms to prepare text for BM25 indexing and querying.
def _tokenize(text: str, mode: str = "simple"):
    ''' 
    Input example: text="Tokyo hosted the G7 Summit", mode="simple"
    Output example: ["tokyo","hosted","the","g7","summit"]
    '''
    if mode == "whitespace": #Whitespace tokenization mode.
        return text.lower().split() #Lowercase and split on spaces.
    import re #Import regex module.
    return re.findall(r"[a-z0-9]+", text.lower()) #Keeping only sequences made by letters and numbers

class BM25Retriever:
    #Function: initialize a BM25 retriever over a corpus.
    def __init__(self, corpus: List[Dict[str, str]], tokenizer: str = "simple"):
        ''' 
        Input example: corpus=[{"text":"Paris is in France"}, {"text":"Rome is in Italy"}]
        Output example: A BM25Retriever object ready to retrieve passages.
        '''
        self.corpus = corpus #Store the original corpus.
        self.tokenizer = tokenizer #Store tokenizer choice.
        self.tokens = [_tokenize(p["text"], tokenizer) for p in corpus] #Tokenize each document text.
        self.bm25 = BM25Okapi(self.tokens) #Build BM25 index over tokenized corpus.

    #Function: retrieve the top-k most relevant passages for a query -> rank corpus passages using BM25 similarity.
    def retrieve(self, query: str, k: int) -> List[Tuple[float, Dict[str, str]]]:
        '''Input example: query="G7 Summit Tokyo", k=5
        Output example: [(score, passage_dict), ...] sorted by relevance.'''
        qtok = _tokenize(query, self.tokenizer) #Tokenize the query
        scores = self.bm25.get_scores(qtok) #Compute BM25 scores for all documents
        # top-k
        import numpy as np #Import numpy for sorting
        idx = np.argsort(scores)[::-1][:k] #Select indices of top-k highest scores
        return [(float(scores[i]), self.corpus[int(i)]) for i in idx] #Return (score, passage) pairs
