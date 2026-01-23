from dataclasses import dataclass, field #Imports dataclass utilities to define configuration objects
from typing import Optional #Imports Optional type for optional configuration fields
import yaml #Imports yaml to load configuration files

@dataclass
class FeverCfg: #Defines configuration options for the FEVER dataset
    enabled: bool = True #Enables or disables the FEVER dataset
    train_jsonl: str = "" #Path to the FEVER training jsonl file
    wiki_pages_dir: str = "" #Path to the FEVER Wikipedia pages directory
    max_examples: int = 500 #Limits the number of FEVER examples used

@dataclass
class HotpotCfg: #Defines configuration options for the HotpotQA dataset
    enabled: bool = True #Enables or disables the HotpotQA dataset
    train_json: str = "" #Path to the HotpotQA training json file
    max_examples: int = 500 #Limits the number of HotpotQA examples used

@dataclass
class DataCfg: #Groups all dataset-related configurations
    fever: FeverCfg #Configuration for the FEVER dataset
    hotpotqa: HotpotCfg #Configuration for the HotpotQA dataset

@dataclass
class RetrievalCfg: #Defines configuration options for the retrieval module
    k: int = 10 #Number of documents or passages retrieved
    use_bm25: bool = True #Enables sparse BM25 retrieval
    use_dense: bool = True #Enables dense embedding-based retrieval
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2" #Model used for dense retrieval embeddings
    bm25_tokenizer: str = "simple" #Tokenizer used for BM25
    max_corpus_sentences: int = 200_000 #Maximum number of sentences indexed from the corpus

@dataclass
class PoisonCfg: #Defines configuration options for data poisoning
    enabled: bool = True #Enables or disables poisoning
    rate: float = 0.2 #Percentage of poisoned data
    strategy: str = "contradictory_passage" #Type of poisoning strategy used
    target: str = "retrieval_set" #Specifies where poisoning is applied

@dataclass
class VerificationCfg: #Defines configuration options for the verification module
    nli_model: str = "roberta-large-mnli" #NLI model used for verification
    contradiction_threshold: float = 0.55 #Threshold to consider contradiction strong
    entailment_threshold: float = 0.55 #Threshold to consider entailment strong
    top_n_verify: int = 5 #Number of top retrieved passages verified with NLI

@dataclass
class GenerationCfg: #Defines configuration options for answer generation
    enabled: bool = False #Enables or disables generation
    provider: str = "hf" #Specifies the generation provider
    hf_model: str = "google/flan-t5-base" #HuggingFace model used for generation
    max_new_tokens: int = 64 #Maximum number of generated tokens
    num_samples: int = 5 #Number of generated samples per query

@dataclass
class EvalCfg: #Defines configuration options for evaluation
    compute_self_consistency: bool = True #Enables self-consistency evaluation

@dataclass
class TrainingNLICfg: #Defines training options for the NLI model
    base_model: str = "roberta-large-mnli" #Base NLI model used for training
    epochs: int = 1 #Number of training epochs
    batch_size: int = 8 #Batch size for NLI training
    lr: float = 2e-5 #Learning rate for NLI training
    max_train_pairs: int = 20_000 #Maximum number of training pairs

@dataclass
class TrainingRetrieverCfg: #Defines training options for the retriever model
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2" #Base retriever model
    epochs: int = 1 #Number of training epochs
    batch_size: int = 32 #Batch size for retriever training
    lr: float = 2e-5 #Learning rate for retriever training
    max_train_pairs: int = 20_000 #Maximum number of training pairs

@dataclass
class TrainingCfg: #Groups all training-related configurations
    enabled: bool = False #Enables or disables training
    output_dir: str = "outputs" #Directory where training outputs are saved
    nli: TrainingNLICfg = field(default_factory=TrainingNLICfg) #NLI training configuration
    retriever: TrainingRetrieverCfg = field(default_factory=TrainingRetrieverCfg) #Retriever training configuration

@dataclass
class RunCfg: #Top-level configuration object for a full experiment run
    seed: int #Random seed for reproducibility
    data: DataCfg #Dataset configuration
    retrieval: RetrievalCfg #Retrieval configuration
    poisoning: PoisonCfg #Poisoning configuration
    verification: VerificationCfg #Verification configuration
    generation: GenerationCfg #Generation configuration
    evaluation: EvalCfg #Evaluation configuration
    training: TrainingCfg #Training configuration

def load_config(path: str) -> RunCfg: #Loads a YAML configuration file and builds a RunCfg object
    with open(path, "r", encoding="utf-8") as f: #Opens the YAML configuration file
        raw = yaml.safe_load(f) #Parses the YAML file into a Python dictionary

    fever = FeverCfg(**raw["data"]["fever"]) #Creates FEVER configuration from YAML
    hotpot = HotpotCfg(**raw["data"]["hotpotqa"]) #Creates HotpotQA configuration from YAML
    data = DataCfg(fever=fever, hotpotqa=hotpot) #Groups dataset configurations

    retrieval = RetrievalCfg(**raw["retrieval"]) #Creates retrieval configuration
    poisoning = PoisonCfg(**raw["poisoning"]) #Creates poisoning configuration
    verification = VerificationCfg(**raw["verification"]) #Creates verification configuration
    generation = GenerationCfg(**raw["generation"]) #Creates generation configuration
    evaluation = EvalCfg(**raw["evaluation"]) #Creates evaluation configuration
    
    t_raw = raw.get("training", {}) or {} #Reads training section if present
    nli_raw = t_raw.get("nli", {}) or {} #Reads NLI training subsection
    ret_raw = t_raw.get("retriever", {}) or {} #Reads retriever training subsection

    training = TrainingCfg(
        enabled=bool(t_raw.get("enabled", False)), #Sets whether training is enabled
        output_dir=str(t_raw.get("output_dir", "outputs")), #Sets training output directory
        nli=TrainingNLICfg(**nli_raw), #Creates NLI training configuration
        retriever=TrainingRetrieverCfg(**ret_raw), #Creates retriever training configuration
    )

    return RunCfg(
        seed=raw["seed"], #Sets the random seed
        data=data, #Assigns dataset configuration
        retrieval=retrieval, #Assigns retrieval configuration
        poisoning=poisoning, #Assigns poisoning configuration
        verification=verification, #Assigns verification configuration
        generation=generation, #Assigns generation configuration
        evaluation=evaluation, #Assigns evaluation configuration
        training=training, #Assigns training configuration
    )
