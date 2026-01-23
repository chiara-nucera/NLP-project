from dataclasses import replace

"""
This file runsan ablation study by changing verification thresholds.
"""

DECISION_RULES = { #Defines different verification decision rules
    "strict": {"entailment_threshold": 0.40, "contradiction_threshold": 0.40}, #Strict rule with higher thresholds
    "balanced": {"entailment_threshold": 0.30, "contradiction_threshold": 0.30}, #Balanced rule with moderate thresholds
    "conservative": {"entailment_threshold": 0.40, "contradiction_threshold": 0.45}, #Conservative rule favoring contradiction
}

#Performing an ablation by varying the verifier’s entailment and contradiction thresholds in 
# the configuration, rebuilding the pipeline for each setting to avoid hidden state.


def run_decision_rule_ablation(cfg, corpus, examples, run_fn, rules=DECISION_RULES, n=200): #Runs ablation over decision rules
    """
    Ablation over verification thresholds stored in cfg.verification:
      - cfg.verification.entailment_threshold
      - cfg.verification.contradiction_threshold
    """ #Explains which parameters are varied in the ablation
    from ragtrust.experiment import RAGTrustPipeline #Imports pipeline locally to avoid side effects

    subset = examples[:n] #Selects a subset of examples for faster ablation
    results = {} #Initializes dictionary to store results

    for name, params in rules.items(): #Loops over each decision rule
        print(f"\n=== ABLATION RULE: {name} ===") #Prints current rule name

        #1. Creatinh a modified cfg for this rule (ONLY thresholds change)
        cfg_rule = replace( #Creates a new configuration object
            cfg,
            verification=replace(
                cfg.verification,
                entailment_threshold=params["entailment_threshold"], #Updates entailment threshold
                contradiction_threshold=params["contradiction_threshold"], #Updates contradiction threshold
            ),
        )

        #2.
        pipeline_rule = RAGTrustPipeline(cfg_rule, corpus) #Builds a new pipeline instance

        #3. Running evaluation
        res = run_fn(cfg_rule, pipeline_rule, subset) #Runs evaluation with the modified rule
        results[name] = res["summary"] #Stores summary results for this rule
        print("Summary:", res["summary"]) #Prints summary results

    return results #Returns results for all decision rules
