#Dense retrieval module: implements a dense retriever using sentence embeddings and FAISS

from typing import List, Dict, Tuple #Import type hints
import numpy as np #Used for array handling and types
import faiss #Library for fast vector similarity search
from sentence_transformers import SentenceTransformer #Model to convert text into embeddings

class DenseRetriever:
    #Function: initialize a dense retriever over a text corpus 
    # -> encode corpus texts into vectors and build a FAISS index.
    def __init__(self, corpus: List[Dict[str, str]], model_name: str):
        self.corpus = corpus #Store the original corpus
        self.model = SentenceTransformer(model_name) #Load sentence embedding model
        texts = [p["text"] for p in corpus] #Extract text field from each passage
        emb = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True
        ) #Encode texts into normalized embeddings
        self.emb = emb.astype("float32") #Convert embeddings to float32 for FAISS
        dim = self.emb.shape[1] #Get embedding dimensionality
        self.index = faiss.IndexFlatIP(dim) #Create FAISS index using inner product
        self.index.add(self.emb) #Add corpus embeddings to the index

    #Function: retrieve top-k most similar passages to the query
    # -> find passages with embeddings closest to the query embedding
    def retrieve(self, query: str, k: int) -> List[Tuple[float, Dict[str, str]]]:
        q = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32") #Encode query into a normalized embedding
        scores, idx = self.index.search(q, k) #Search FAISS index for top-k matches
        out = [] #Collect results
        for s, i in zip(scores[0], idx[0]): #Loop over returned scores and indices
            out.append((float(s), self.corpus[int(i)])) #Attach score to corresponding passage
        return out #Return ranked list of (score, passage)
