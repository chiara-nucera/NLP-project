#Data preparation
#Loading HotpotQA examples (questions + context paragraphs) and building a sentence-level corpus
#that can be used by retrieval and verification components in the RAG pipeline

"""
This file loads and preprocesses the HotpotQA dataset. It supports both JSON (a list of
examples) and JSONL (one JSON object per line) input formats. The loader converts raw
examples into structured HotpotExample objects, normalizes Wikipedia titles, and
optionally builds a sentence-level corpus from the provided "context" field. This module
does not perform retrieval, prediction, or training; it only prepares the data required
by downstream components.
"""
'''
HotpotQA is a multi-hop question answering dataset where each example includes a question, 
a set of Wikipedia paragraphs as context, and annotated supporting facts. In this project, 
it is used to evaluate reasoning and verification in a RAG pipeline.
'''

import json #to parse JSON/JSONL files
from dataclasses import dataclass #to define a lightweight structured container
from typing import List, Dict, Any, Tuple, Optional #to annotate expected data structures
from ragtrust.utils import norm_title #to normalize titles (e.g., spaces->underscores)


@dataclass
#Defining a structured representation for a single HotpotQA example
class HotpotExample:
    id: str #unique identifier of the example
    question: str #question to be answered
    answer: str #gold answer text (may be empty in some splits)
    supporting_facts: List[Tuple[str, int]] #gold supporting facts as (title, sentence_index)
    context: List[Tuple[str, List[str]]] #provided context as (title, list_of_sentences)
    qtype: Optional[str] = None #optional question type metadata


#Loading HotpotQA examples and supporting both JSON and JSONL input formats
def load_hotpot_examples(json_path: str, max_examples: int) -> List[HotpotExample]:
    """
    Supports:
    - JSON (list of examples)  -> .json
    - JSONL (one json per line) -> .jsonl
    """
    #Reading the first two lines to infer whether the file is JSON or JSONL
    with open(json_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
        second = f.readline().strip()

    #For JSONL detection:
    #If the first line looks like a JSON object and the file has more non-empty lines, it is likely JSONL
    looks_like_json_object = first.startswith("{") and first.endswith("}")
    has_more_lines = bool(second)

    is_jsonl = json_path.lower().endswith(".jsonl") or (looks_like_json_object and has_more_lines)

    #JSONL path
    #For each JSON line converting it in a dictionary, extracting the consext (provided by the dataset)
    #
    if is_jsonl:
        out: List[HotpotExample] = [] #list collecting parsed examples
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue #skipping empty lines
                ex = json.loads(line) #parsing one JSON object

                #Parsing the provided context: list of [title, [sentences...]]
                #Each element of the context is a Wikip. page => considering the page title and the list of sentences 
                # + normalizing the title
                ctx = []
                for c in ex.get("context", []):
                    title = c[0]
                    sents = c[1]
                    ctx.append((norm_title(title), sents)) #normalizing titles for consistency

                #Extracting gold evidences as (title, sent_idx) (Wikip. title and index of the sentence)
                #If the sentence index is not numeric, we store -1 to avoid errors
                supp = [(norm_title(sf[0]), int(sf[1]) if str(sf[1]).isdigit() else -1)
                        for sf in ex.get("supporting_facts", [])]

                #Building the structured HotpotExample object
                out.append(HotpotExample( #Adding the element to the list
                    id=str(ex.get("_id", ex.get("id", ""))), #handling alternative id field names
                    question=ex["question"],
                    answer=ex.get("answer", ""),
                    supporting_facts=supp,
                    context=ctx,
                    qtype=ex.get("type", None),
                ))
                
                if len(out) >= max_examples:
                    break #limiting number of examples
        return out

    #JSON list path
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f) #loading the entire list of examples HotpotQA

    out: List[HotpotExample] = [] #list collecting parsed examples
    for ex in data[:max_examples]: #taking only the first max_examples to control the dataset dimension
        #The context is returned: list of [title, [sentences...]] -> title and list of sentences
        #Everything is saved in ctx
        ctx = []
        for c in ex.get("context", []):
            title = c[0]
            sents = c[1]
            ctx.append((norm_title(title), sents)) #normalizing titles for consistency

        #Parsing gold supporting facts as (title, sent_idx)
        supp = [(norm_title(sf[0]), int(sf[1]) if str(sf[1]).isdigit() else -1) #if index is not valid -> -1
                for sf in ex.get("supporting_facts", [])] 

        #Building the list of HotpotExample
        out.append(HotpotExample( 
            id=str(ex.get("_id", ex.get("id", ""))), #handling alternative id field names
            question=ex["question"],
            answer=ex.get("answer", ""),
            supporting_facts=supp,
            context=ctx,
            qtype=ex.get("type", None),
        ))
    return out


#HotpotQA gave the context (pages + sentences), now we have to transform it in a corpus of sentences for the RAG
#{doc_id,title,sent_id,text}

#Building a sentence-level corpus from HotpotQA "context" paragraphs
def build_hotpot_corpus(examples):
    corpus = []
    for ex in examples:
        for title, sentences in ex.context:
            for i, sent in enumerate(sentences):
                corpus.append({
                    "title": title,
                    "text": sent,
                    "doc_id": f"{title}__{i}",  
                    "source": "hotpot"          
                })
    return corpus

