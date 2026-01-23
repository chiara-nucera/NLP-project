"""
This file defines the main RAGTrust pipeline.
It combines retrieval, optional poisoning, verification with NLI,
optional generation, and returns all intermediate results needed
for evaluation and analysis.
"""
from typing import Dict, Any, List #Imports typing utilities for dictionaries and lists
from ragtrust.config import RunCfg #Imports the global run configuration
from ragtrust.indexing.hybrid import HybridRetriever #Imports the hybrid retriever (BM25 + dense)
from ragtrust.poisoning.strategies import Poisoner #Imports the poisoning module
from ragtrust.verification.nli import NLIModel #Imports the NLI model for verification
from ragtrust.verification.verifier import MultiSourceVerifier #Imports the multi-source verifier
from ragtrust.generation.prompts import build_meta_prompt #Imports prompt builder for generation
from ragtrust.generation.llm import HFGenerator #Imports HuggingFace-based generator
from ragtrust.evaluation.metrics import ( #Imports evaluation metrics
    fever_accuracy, hotpot_exact_match, hallucination_proxy, self_consistency_score
)

class RAGTrustPipeline: #Defines the main RAG pipeline class
    def __init__(self, cfg: RunCfg, corpus: List[Dict[str, str]]): #Initializes the pipeline with config and corpus
        self.cfg = cfg #Stores the run configuration
        
        # 1) crea prima il poisoner (serve anche per corpus poisoning)
        self.poisoner = Poisoner( #Creates the poisoner object
            strategy=cfg.poisoning.strategy, #Sets the poisoning strategy
            rate=cfg.poisoning.rate if cfg.poisoning.enabled else 0.0, #Sets poisoning rate or zero if disabled
            seed=cfg.seed, #Uses seed for reproducibility
        )

        # 2) corpus poisoning (solo se target == "corpus")
        if self.cfg.poisoning.enabled and self.cfg.poisoning.target == "corpus": #Checks if corpus poisoning is enabled
            corpus = self.poisoner.apply_to_corpus(corpus) #Applies poisoning directly to the corpus

        # 3) poi costruisci il retriever sul corpus (pulito o poisonato)
        self.retriever = HybridRetriever( #Builds the hybrid retriever
            corpus=corpus, #Uses the (possibly poisoned) corpus
            use_bm25=cfg.retrieval.use_bm25, #Enables or disables BM25 retrieval
            use_dense=cfg.retrieval.use_dense, #Enables or disables dense retrieval
            dense_model=cfg.retrieval.dense_model, #Sets the dense embedding model
            bm25_tokenizer=cfg.retrieval.bm25_tokenizer, #Sets the BM25 tokenizer
        )

        self.nli = NLIModel(cfg.verification.nli_model) #Initializes the NLI model
        self.verifier = MultiSourceVerifier( #Initializes the multi-source verifier
            nli=self.nli, #Uses the NLI model
            thr_con=cfg.verification.contradiction_threshold, #Sets contradiction threshold
            thr_ent=cfg.verification.entailment_threshold, #Sets entailment threshold
        )
        self.generator = HFGenerator(cfg.generation.hf_model, cfg.generation.max_new_tokens) if cfg.generation.enabled else None #Initializes generator only if enabled

    def run_one(self, claim: str, verifier_mode: str = "single", verbose: bool = False): #Runs the pipeline on a single claim
        retrieved = self.retriever.retrieve(claim, k=self.cfg.retrieval.k) #Retrieves top-k passages for the claim
        #apply poisoning only if the target is the retrieved set 
        poisoned = retrieved #Initializes poisoned set as retrieved set
        if self.cfg.poisoning.enabled and self.cfg.poisoning.target == "retrieval_set": #Checks if retrieval-set poisoning is enabled
            poisoned = self.poisoner.apply(retrieved) #Applies poisoning to retrieved passages
      
        #DEBUG: check that score survived poisoning
        if verbose:
            missing_score = sum(1 for p in poisoned if "score" not in p)
            print("Missing score after poisoning:", missing_score, "/", len(poisoned))
            if len(poisoned) > 0:
                print("Example keys:", list(poisoned[0].keys()))

        injected = sum(1 for p in poisoned if p.get("is_poison")) #Counts how many poisoned passages were injected
        if verbose:
            print("Injected poison:", injected) #Prints number of poisoned passages if verbose
        
        #Without top ranking:
        #to_verify = poisoned[: self.cfg.verification.top_n_verify] #Selects top passages to verify
        
        #With top ranking: 
        #RERANK after poisoning/filtering: keep highest-scoring passages first
        def _score(p):
            try:
                return float(p.get("score", 0.0))
            except (TypeError, ValueError):
                return 0.0

        ranked = sorted(poisoned, key=_score, reverse=True)
        n = min(self.cfg.verification.top_n_verify, len(ranked))
        to_verify = ranked[:n]

        if verbose:
            print("Top reranked scores:", [d.get("score", None) for d in to_verify[:5]])
        
        verification = self.verifier.analyze(claim, to_verify, mode=verifier_mode) #Runs NLI-based verification
        if verbose:
            print("N evidences verified:", len(to_verify)) #Prints number of verified evidences

        generations = [] #Initializes empty list for generations
        if self.generator is not None: #Checks if generation is enabled
            prompt = build_meta_prompt(claim, to_verify) #Builds the generation prompt
            generations = self.generator.generate(prompt, num_samples=self.cfg.generation.num_samples) #Generates answers

        return { #Returns all intermediate and final outputs
            "claim": claim, #Original claim
            "retrieved": retrieved, #Retrieved passages
            "poisoned": poisoned, #Poisoned passages
            "verification": verification, #Verification results
            "generations": generations, #Generated answers
        }
    
    def predict_label_from_verification(self, verification: dict) -> str: #Predicts a final label from verification signals
    # Use ablation-provided params if present, otherwise fall back to defaults (balanced-like)
        params = getattr( #Reads decision parameters if provided
            self,
            "decision_params",
            {"min_strength": 0.3, "margin": 0.10, "conflict_nei": True}, #Default decision parameters
        )

        # Read aggregated verification signals
        s_ent = float(verification.get("support_strength", 0.0)) #Reads entailment strength
        s_con = float(verification.get("contradiction_strength", 0.0)) #Reads contradiction strength
        conflict = bool(verification.get("conflict", False)) #Checks if evidences conflict

        # Unpack decision parameters
        min_strength = float(params.get("min_strength", 0.3)) #Minimum strength to make a decision
        margin = float(params.get("margin", 0.10)) #Required margin between signals
        conflict_nei = bool(params.get("conflict_nei", True)) #Whether conflicts force NEI

        # If evidences conflict and we are conservative about conflicts, return NEI
        if conflict and conflict_nei:
            return "NOT ENOUGH INFO" #Returns NEI if evidences disagree

        # Decide only if one side is strong enough AND sufficiently larger than the other
        if s_con >= min_strength and (s_con - s_ent) > margin:
            return "REFUTES" #Returns REFUTES if contradiction dominates

        if s_ent >= min_strength and (s_ent - s_con) > margin:
            return "SUPPORTS" #Returns SUPPORTS if entailment dominates

        # Otherwise, not enough evidence to decide
        return "NOT ENOUGH INFO" #Returns NEI if no clear decision is possible
