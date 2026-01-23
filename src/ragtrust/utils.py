"""
This file defines small utility functions.
"""
import random #Imports Python random module
import numpy as np #Imports NumPy for numerical operations
import torch #Imports PyTorch for deep learning utilities
import re #Imports regular expressions module

def set_seed(seed: int) -> None: #Sets all random seeds for reproducibility
    random.seed(seed) #Fixes Python random seed
    np.random.seed(seed) #Fixes NumPy random seed
    torch.manual_seed(seed) #Fixes PyTorch CPU random seed
    torch.cuda.manual_seed_all(seed) #Fixes PyTorch GPU random seed

_TITLE_CLEAN = re.compile(r"\s+|_") #Regex to match spaces and underscores

def norm_title(t: str) -> str: #Normalizes a title string to a standard format
    t = t.strip() #Removes leading and trailing spaces
    t = t.replace(" ", "_") #Replaces spaces with underscores
    t = _TITLE_CLEAN.sub("_", t) #Collapses multiple spaces or underscores into one
    return t #Returns the normalized title

def simple_sentence_split(text: str): #Splits text into sentences in a simple way
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip()) #Splits text after punctuation marks
    return [p.strip() for p in parts if p.strip()] #Returns cleaned non-empty sentences
