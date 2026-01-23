#Retriever training module
#Builds bi-encoder training pairs from FEVER and fine-tunes a SentenceTransformer retriever

import os #Path utilities
import random #Deterministic shuffling and sampling
from typing import List, Dict, Any, Tuple #Type hints

from sentence_transformers import SentenceTransformer, InputExample, losses #Bi-encoder model, training examples, and loss
from torch.utils.data import DataLoader #Mini-batch loader

from ragtrust.utils import norm_title #Title normalization helper
from ragtrust.training.fever_pairs import _build_sentence_lookup, _extract_gold_sentence_keys #Reuse FEVER evidence helpers

#Function: create training pairs for a bi-encoder retriever
'''
Input example #fever_examples contains claim label evidence, max_pairs=1000 seed=42
Output example #List of InputExample where texts=[claim, gold_sentence]
'''
def make_retriever_pairs(fever_examples, corpus, max_pairs: int, seed: int) -> List[InputExample]:
    """
    Build training pairs for bi-encoder
    query=claim
    positive=gold evidence sentence for SUPPORTS or REFUTES
    """
    rng = random.Random(seed) #Deterministic RNG
    lookup = _build_sentence_lookup(corpus) #Map (title sent_id) to sentence text

    pairs: List[InputExample] = [] #Collected training pairs
    for ex in fever_examples: #Loop over FEVER examples
        if ex.label not in ("SUPPORTS", "REFUTES"): #Skip NEI examples
            continue
        keys = _extract_gold_sentence_keys(ex.evidence) #Get gold evidence keys
        for (t, sid) in keys: #Loop over gold evidence keys
            prem = lookup.get((t, sid)) #Resolve sentence text from corpus
            if prem: #If sentence exists
                pairs.append(InputExample(texts=[ex.claim, prem])) #Add (claim, positive evidence) pair
                if len(pairs) >= max_pairs: #Stop early if max reached
                    return pairs
    rng.shuffle(pairs) #Shuffle pairs for training
    return pairs #Return list of training examples

#Function: fine-tune a SentenceTransformer retriever on FEVER pairs
'''
Input example: base_model="all-MiniLM-L6-v2" output_dir="models/retriever" epochs=1 batch_size=64 lr=2e-5 seed=42
Output example: path to the saved fine-tuned retriever directory
'''
def train_retriever_on_fever(
    fever_examples,
    fever_corpus,
    base_model: str,
    output_dir: str,
    max_train_pairs: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> str:
    model = SentenceTransformer(base_model) #Load base bi-encoder model

    train_examples = make_retriever_pairs(
        fever_examples=fever_examples,
        corpus=fever_corpus,
        max_pairs=max_train_pairs,
        seed=seed,
    ) #Build training examples from FEVER

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size) #Create batches of pairs
    train_loss = losses.MultipleNegativesRankingLoss(model) #Loss where in-batch positives act as negatives

    warmup_steps = int(0.1 * len(train_loader) * epochs) #Warmup steps as 10 percent of total steps

    model.fit(
        train_objectives=[(train_loader, train_loss)], #Training data and loss
        epochs=epochs, #Training epochs
        warmup_steps=warmup_steps, #LR warmup
        optimizer_params={"lr": lr}, #Learning rate
        show_progress_bar=True, #Show training progress
    ) #Run fine-tuning

    model.save(output_dir) #Save trained retriever
    return output_dir #Return saved path
