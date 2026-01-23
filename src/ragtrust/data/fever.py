#Data preparation
#Transforming the dataset FEVER into structure examples (one line is one example) and a corpus (Wikipedia sentences) that the RAG can use

"""
This file handles loading and preprocessing of the FEVER dataset and the associated
Wikipedia pages. It converts raw FEVER JSONL examples into structured objects and builds
a sentence-level Wikipedia corpus, which is later used by retrieval and
verification components. The module does not perform retrieval or prediction, but
provides the ground-truth data representation required by the RAG pipeline.
"""


import json #to read the json file 
from dataclasses import dataclass 
from typing import List, Dict, Any, Iterable, Optional, Set, Tuple #to define data structures
from pathlib import Path #to deal with the path on filesystem
from ragtrust.utils import norm_title, simple_sentence_split #imports norm_title (to normalize Wikipedia titles (spaces -> _))
                                                             #and simple_sentence_split (dividing the text in sentences)

@dataclass
#Defining a data structure for a single example Fever so that they are readable
class FeverExample:
    id: int #ID of the claim in the dataset
    claim: str #claim to be verified
    label: str  # SUPPORTS / REFUTES / NOT ENOUGH INFO (correct claim)
    evidence: Any #contains the evidences 

#reading Fever file and returning a list of FeverExample (each element is a dataclass with id, claim, label, evidence)
def load_fever_examples(jsonl_path: str, max_examples: int) -> List[FeverExample]:
    out: List[FeverExample] = [] #to return the list of fever example
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line) #converting the JSON line in a dictionary
            out.append(FeverExample(  #Creating the FeverExample object and adding it to the list
                id=int(ex["id"]),
                claim=ex["claim"],
                label=ex["label"],
                evidence=ex.get("evidence", None),
            ))
            if len(out) >= max_examples:
                break
    return out

#Extracting the titles in the gold evidences to know which pages the dataset considers correct for each claim
#The output is a set with strings (no duplicates)
def extract_gold_titles(evidence_field: Any) -> Set[str]:
    titles: Set[str] = set() #Using set to avoid duplicates
    if not evidence_field: #If there are no evidences -> empty set
        return titles
    for ev_set in evidence_field: #iterating on evidence sets
        for item in ev_set: #iterating on sigle evidences
            #checking that the item exists and is not empty or null
            #if both are true: the title is true and can be considered a gold evidence
            #bc each item is a list like [ann_id, ev_id, wiki_title, sentence_id]
            if len(item) >= 3 and item[2]: 
                #adding the (normalized) title to the set
                titles.add(norm_title(str(item[2])))
    return titles #returns the gold titles

def iter_wiki_pages(wiki_pages_dir: str) -> Iterable[Dict[str, Any]]:
    """
    wiki-pages/*.jsonl each row is a line
    {id: page_name, text: ..., lines: ...}
    """
    p = Path(wiki_pages_dir) #Creating a path object for the directory
    for fp in sorted(p.glob("*.jsonl")): #iterating on the jsonl 
        with open(fp, "r", encoding="utf-8") as f: #opening each file
            for line in f: #each line is a Wikip. page
                yield json.loads(line) #returning the page

#Function to return the corpus of sentences used by the retriever
def build_fever_corpus(
    wiki_pages_dir: str, #directory with Wikipedia pages
    wanted_titles: Optional[Set[str]] = None, #Filtering to have only gold eviedences
    max_sentences: int = 200_000, #Limit the number of sentences in the corpus
) -> List[Dict[str, str]]: #list of the corpus
    """
    Each element: {doc_id, title, sent_id, text}
    """
    corpus: List[Dict[str, str]] = []
    n = 0 #counter for sentences 
    for page in iter_wiki_pages(wiki_pages_dir): #iterating on the Wikipedia pages
        title = norm_title(page.get("id", "")) #Extracting and normalizing the title of the page
        if wanted_titles is not None and title not in wanted_titles: #Skipping the non-gold titles
            continue

        # Case 1: pages with lines
        #Checking if the page containt numbered sentences
        if "lines" in page and isinstance(page["lines"], str) and page["lines"].strip():
            # some versions: "0\tSentence...\n1\tSentence..."
            for row in page["lines"].split("\n"):
                #Each line is a sentence
                row = row.strip()
                if not row:
                    continue
                #Skipping empty lines
                if "\t" in row:
                    sid, sent = row.split("\t", 1)
                else:
                    sid, sent = "0", row
                sent = sent.strip()
                if sent:
                    corpus.append({ #Adding the sentence to the corpus
                        "doc_id": title,
                        "title": title,
                        "sent_id": str(sid),
                        "text": sent,
                    })
                    n += 1
                    if n >= max_sentences: #Stopping if max reached
                        return corpus
        else:
            #Case 2: fallback: split sul testo
            #If there are no lines, the entire text is used
            text = page.get("text", "")
            for i, sent in enumerate(simple_sentence_split(text)): #Dividing the text in sentences
                corpus.append({ #Adding each sentence
                    "doc_id": title,
                    "title": title,
                    "sent_id": str(i),
                    "text": sent,
                })
                n += 1
                if n >= max_sentences:
                    return corpus
    return corpus #Returning the entire corpus
