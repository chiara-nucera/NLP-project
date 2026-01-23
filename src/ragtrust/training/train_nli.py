#NLI training module#Fine-tunes a sequence classification model on FEVER-derived NLI pairs

import os #Used for filesystem paths
from dataclasses import asdict #Utility for dataclass conversion
from typing import List, Dict, Any #Import type hints

import torch #Used to check GPU availability
from datasets import Dataset #HuggingFace dataset wrapper
from transformers import (
    AutoTokenizer, #Tokenizer loader
    AutoModelForSequenceClassification, #Sequence classification model
    TrainingArguments, #Training configuration
    Trainer, #High-level training API
)

from ragtrust.training.fever_pairs import make_fever_nli_pairs #Build FEVER NLI pairs

#Function: fine-tune an NLI model on FEVER data
'''
Input example: base_model="roberta-large-mnli", max_train_pairs=5000, epochs=3
Output example: path to the saved fine-tuned model directory
'''
def train_nli_on_fever(
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
    """
    Fine-tune NLI model on FEVER
    premise=evidence sentence hypothesis=claim
    """
    pairs = make_fever_nli_pairs(
        fever_examples=fever_examples,
        corpus=fever_corpus,
        max_pairs=max_train_pairs,
        seed=seed,
    ) #Create NLI training pairs from FEVER

    ds = Dataset.from_list(pairs) #Convert list of pairs into HuggingFace dataset

    tok = AutoTokenizer.from_pretrained(base_model) #Load tokenizer from base model
    model = AutoModelForSequenceClassification.from_pretrained(base_model) #Load NLI model

    #Function: tokenize premise-hypothesis pairs
    '''
    Input example#{"premise":"Paris is in France","hypothesis":"Paris is in Europe"}
    Output example#Token ids and attention masks
    '''
    def tokenize(batch):
        return tok(
            batch["premise"], #Premise input
            batch["hypothesis"], #Hypothesis input
            truncation=True, #Cut sequences if too long
            padding="max_length", #Pad all sequences equally
            max_length=256, #Maximum sequence length
        )

    ds = ds.map(tokenize, batched=True) #Apply tokenization to full dataset
    ds = ds.train_test_split(test_size=0.1, seed=seed) #Split into train and test

    args = TrainingArguments(
        output_dir=output_dir, #Where to save checkpoints
        num_train_epochs=epochs, #Number of training epochs
        per_device_train_batch_size=batch_size, #Train batch size
        per_device_eval_batch_size=batch_size, #Eval batch size
        learning_rate=lr, #Optimizer learning rate
        evaluation_strategy="epoch", #Evaluate once per epoch
        save_strategy="epoch", #Save model once per epoch
        logging_steps=50, #Log every N steps
        seed=seed, #Random seed
        fp16=torch.cuda.is_available(), #Use mixed precision if GPU supports it
        report_to="none", #Disable external logging
    )

    trainer = Trainer(
        model=model, #Model to train
        args=args, #Training configuration
        train_dataset=ds["train"], #Training split
        eval_dataset=ds["test"], #Evaluation split
        tokenizer=tok, #Tokenizer used for padding and decoding
    )
    trainer.train() #Run fine-tuning

    trainer.save_model(output_dir) #Save trained model weights
    tok.save_pretrained(output_dir) #Save tokenizer files

    return output_dir #Return path to saved model
