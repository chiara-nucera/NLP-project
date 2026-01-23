#Experiment execution
#Running the full RAG pipeline on FEVER and HotpotQA and collecting evaluation results

"""
This file runs the RAGTrust pipeline on a set of examples, collects per-example predictions,
computes evaluation metrics, and aggregates the results at the dataset level.
The logic is shared across datasets, with small differences between FEVER and HotpotQA.
"""

from typing import Dict, Any, List #to annotate expected input/output structures
from tqdm import tqdm #to display a progress bar during experiments

from ragtrust.config import RunCfg #experiment configuration
from ragtrust.experiment import RAGTrustPipeline #main RAG pipeline
from ragtrust.evaluation.metrics import (
    fever_accuracy, hotpot_exact_match, hallucination_proxy, self_consistency_score
) #evaluation metrics
from ragtrust.evaluation.runner import eval_fever_batch, eval_hotpot_batch #batch aggregation


#Running the experiment on the FEVER dataset
def run_fever(cfg: RunCfg, pipeline: RAGTrustPipeline, examples) -> Dict[str, Any]:
    #cfg = configuration of the experiment
    #pipeline = RAG system
    #example = list of FEVER examples (claim + label)
    '''
    Input: config + model + data
    Output: {
    "summary": {...},  #metrics (accuracy, hallucination, ecc.)
    "rows": [...]      #list of results for each example
    }
    '''
    rows = [] #list collecting per-example evaluation results

    for ex in tqdm(examples, desc="FEVER", leave=False, disable=True): #desc="FEVER" -> label of the progress bar
        #Running the RAG pipeline on the claim
        #verifier_mode="single" means evidence is verified independently
        out = pipeline.run_one(ex.claim, verifier_mode="single")

        #Predicting the FEVER label (SUPPORTS / REFUTES / NEI) from verification results 
        # ((out["verification"]) is the output of the NLI model)
        pred = pipeline.predict_label_from_verification(out["verification"]) 

        #Storing per-example results and metrics
        row = {
            "id": ex.id, #example identifier
            "pred": pred, #predicted label
            "gold": ex.label, #gold label from the dataset
            "acc": fever_accuracy(pred, ex.label), #correctness of the prediction
            "hallucination": hallucination_proxy(out["verification"]), #hallucination risk
        }

        #Optionally computing self-consistency if multiple generations are available
        if cfg.evaluation.compute_self_consistency and out["generations"]:
            row["self_consistency"] = self_consistency_score(out["generations"])

        rows.append(row)

    #Returning both aggregated metrics and per-example details
    return {"summary": eval_fever_batch(rows), "rows": rows}

def hotpot_yesno_from_verifier_label(label: str) -> str:
    if label == "SUPPORTS":
        return "yes"
    if label == "REFUTES":
        return "no"
    return "unknown"

#Running the experiment on the HotpotQA dataset
def run_hotpot(cfg: RunCfg, pipeline: RAGTrustPipeline, examples) -> Dict[str, Any]:
    rows = [] #list collecting per-example evaluation results

    for ex in tqdm(examples, desc="HotpotQA", leave=False, disable=True):
        #Using the question as query
        #verifier_mode="multi" allows multi-hop reasoning over multiple evidences
        
        out = pipeline.run_one(ex.question, verifier_mode="multi") #Reasoning on more than 1 sentences/pages
        
                # CORE mode: if generation is off, derive yes/no from verifier label
        if not cfg.generation.enabled:
            ver_label = pipeline.predict_label_from_verification(out["verification"])
            pred_answer = hotpot_yesno_from_verifier_label(ver_label)
        else:
            # existing generation-based majority vote
            pred_answer = ""
            if out["generations"]:
                pred_answer = max(
                    [g.strip() for g in out["generations"]],
                    key=lambda s: sum(s == x for x in [g.strip() for g in out["generations"]])
                )

        #Selecting a single predicted answer from multiple generations
        #Here we use the most frequent answer (self-consistency majority)
        pred_answer = ""
        if out["generations"]:
            pred_answer = max(
                [g.strip() for g in out["generations"]],
                key=lambda s: sum(s == x for x in [g.strip() for g in out["generations"]])
            )
        else:
            #Fallback when no generator is used
            pred_answer = ""

        #Storing per-example results and metrics
        row = {
            "id": ex.id, #example identifier
            "pred": pred_answer, #predicted answer
            "gold": ex.answer, #gold answer from the dataset
            "acc": hotpot_exact_match(pred_answer, ex.answer), #answer correctness
            "hallucination": hallucination_proxy(out["verification"]), #hallucination risk
        }

        #Optionally computing self-consistency if multiple generations are available
        if cfg.evaluation.compute_self_consistency and out["generations"]:
            row["self_consistency"] = self_consistency_score(out["generations"])

        rows.append(row)

    #Returning both aggregated metrics and per-example details
    return {"summary": eval_hotpot_batch(rows), "rows": rows}




