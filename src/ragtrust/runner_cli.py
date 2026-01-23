"""
This file loads the configuration, datasets, optional training,
builds the RAGTrust pipeline, and runs evaluation on FEVER and HotpotQA.
"""
import os #Imports utilities for file paths and directories
from ragtrust.config import load_config #Imports function to load YAML configuration
from ragtrust.utils import set_seed #Imports function to fix random seeds

from ragtrust.data.fever import load_fever_examples, extract_gold_titles, build_fever_corpus #Imports FEVER data utilities
from ragtrust.data.hotpotqa import load_hotpot_examples, build_hotpot_corpus #Imports HotpotQA data utilities

from ragtrust.training.train_nli import train_nli_on_fever #Imports NLI training function
from ragtrust.training.train_retriever import train_retriever_on_fever #Imports retriever training function

from ragtrust.experiment import RAGTrustPipeline #Imports the main RAG pipeline
from ragtrust.evaluation.run_experiment import run_fever, run_hotpot #Imports evaluation functions
import argparse #Imports argument parser for command-line execution

def main(): #Defines the main execution function
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) #Computes project root directory
    os.chdir(PROJECT_ROOT) #Sets working directory to project root

    #cfg = load_config("configs/default.yaml")
   
    parser = argparse.ArgumentParser() #Creates argument parser
    parser.add_argument("--config", type=str, default="configs/default.yaml") #Adds config file argument
    args = parser.parse_args() #Parses command-line arguments

    cfg = load_config(args.config) #Loads configuration from YAML file

    set_seed(cfg.seed) #Fixes random seed for reproducibility

    #FEVER load + corpus
    fever_ex, corpus_fever = None, None #Initializes FEVER examples and corpus
    if cfg.data.fever.enabled: #Checks if FEVER is enabled
        fever_ex = load_fever_examples(cfg.data.fever.train_jsonl, cfg.data.fever.max_examples) #Loads FEVER examples

        wanted = set() #Initializes set of required Wikipedia titles
        for ex in fever_ex:
            wanted |= extract_gold_titles(ex.evidence) #Collects gold evidence titles from FEVER data

        corpus_fever = build_fever_corpus( #Builds FEVER corpus
            cfg.data.fever.wiki_pages_dir, #Wikipedia pages directory
            wanted_titles=wanted, #Only keeps relevant pages
            max_sentences=cfg.retrieval.max_corpus_sentences, #Limits corpus size
        )

    #TRAINING (if enabled)
    if cfg.training.enabled and fever_ex is not None and corpus_fever is not None: #Checks if training is enabled
        out_root = cfg.training.output_dir #Reads training output directory
        os.makedirs(out_root, exist_ok=True) #Creates output directory if missing

        # 1) NLI fine-tuning
        nli_out = os.path.join(out_root, "nli_fever_ft") #Defines NLI output path
        train_nli_on_fever( #Runs NLI fine-tuning on FEVER
            fever_examples=fever_ex, #Uses FEVER examples
            fever_corpus=corpus_fever, #Uses FEVER corpus
            base_model=cfg.training.nli.base_model, #Base NLI model
            output_dir=nli_out, #Output directory
            max_train_pairs=cfg.training.nli.max_train_pairs, #Limits training pairs
            epochs=cfg.training.nli.epochs, #Training epochs
            batch_size=cfg.training.nli.batch_size, #Training batch size
            lr=cfg.training.nli.lr, #Learning rate
            seed=cfg.seed, #Random seed
        )
        cfg.verification.nli_model = nli_out #Updates config to use trained NLI model

        # 2) Retriever fine-tuning
        retr_out = os.path.join(out_root, "retriever_fever_ft") #Defines retriever output path
        train_retriever_on_fever( #Runs retriever fine-tuning on FEVER
            fever_examples=fever_ex, #Uses FEVER examples
            fever_corpus=corpus_fever, #Uses FEVER corpus
            base_model=cfg.training.retriever.base_model, #Base retriever model
            output_dir=retr_out, #Output directory
            max_train_pairs=cfg.training.retriever.max_train_pairs, #Limits training pairs
            epochs=cfg.training.retriever.epochs, #Training epochs
            batch_size=cfg.training.retriever.batch_size, #Training batch size
            lr=cfg.training.retriever.lr, #Learning rate
            seed=cfg.seed, #Random seed
        )
        cfg.retrieval.dense_model = retr_out #Updates config to use trained retriever

    #EVAL FEVER
        #Run on a subset for quick iteration
        if cfg.data.fever.enabled and fever_ex is not None and corpus_fever is not None: #Checks if FEVER evaluation is possible
            pipeline_fever = RAGTrustPipeline(cfg, corpus_fever) #Builds FEVER pipeline
            fever_res = run_fever(cfg, pipeline_fever, fever_ex) #Runs FEVER evaluation
            print("FEVER summary:", fever_res["summary"]) #Prints FEVER results
            
        print("Training enabled:", cfg.training.enabled) #Prints training status
        print("Poisoning enabled:", cfg.poisoning.enabled) #Prints poisoning status
        print("Poisoning rate:", cfg.poisoning.rate) #Prints poisoning rate
        print("Retrieval k:", cfg.retrieval.k) #Prints retrieval k value

    #HotpotQA
    if cfg.data.hotpotqa.enabled: #Checks if HotpotQA is enabled
        hotpot_ex = load_hotpot_examples(cfg.data.hotpotqa.train_json, cfg.data.hotpotqa.max_examples) #Loads HotpotQA examples
        corpus_hotpot = build_hotpot_corpus(hotpot_ex) #Builds HotpotQA corpus

        pipeline_hotpot = RAGTrustPipeline(cfg, corpus_hotpot) #Builds HotpotQA pipeline
        hotpot_res = run_hotpot(cfg, pipeline_hotpot, hotpot_ex) #Runs HotpotQA evaluation
        print("HotpotQA summary:", hotpot_res["summary"]) #Prints HotpotQA results

if __name__ == "__main__":
    main() #Runs main when the script is executed
