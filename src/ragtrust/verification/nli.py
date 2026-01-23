#NLI scoring module#Wraps an MNLI model to score entailment contradiction neutral between evidence and claim

from typing import List, Dict #Type hints
import torch #Tensor operations and device handling
from transformers import AutoTokenizer, AutoModelForSequenceClassification #HuggingFace tokenizer and model loader

class NLIModel:
    """
    MNLI wrapper
    premise=evidence sentence
    hypothesis=claim
    outputs contradiction neutral entailment scores
    """
    #Function: load tokenizer and model and move it to CPU or GPU
    '''
    Input example model_name="roberta-large-mnli"
    Output example An NLIModel object ready to score pairs
    '''
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name) #Load tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name) #Load MNLI classifier
        self.model.eval() #Set model to evaluation mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu" #Pick device
        self.model.to(self.device) #Move model to device

        #Label id mapping used by roberta-large-mnli
        self.contrad_id = 0 #Index for contradiction
        self.neutral_id = 1 #Index for neutral
        self.entail_id = 2 #Index for entailment

    
    #Function: score (premise hypothesis) and return MNLI probabilities
        ''' Input example premise="Paris is in France" hypothesis="Paris is in Europe"
        Output example {"contradiction":0.01,"neutral":0.10,"entailment":0.89} '''
    @torch.inference_mode() #Disable gradients for faster inference
    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        enc = self.tok(
            premise,
            hypothesis,
            truncation=True,
            return_tensors="pt"
        ).to(self.device) #Tokenize pair and move to device
        logits = self.model(**enc).logits[0] #Forward pass to get logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist() #Convert logits to probabilities on CPU
        return {
            "contradiction": float(probs[self.contrad_id]), #Probability of contradiction
            "neutral": float(probs[self.neutral_id]), #Probability of neutral
            "entailment": float(probs[self.entail_id]), #Probability of entailment
        } #Return probabilities
