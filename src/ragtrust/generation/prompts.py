#Prompt construction module: Builds the prompt given to the LLM using the claim and retrieved passages.

#Function: Build the final prompt given to the LLM using the claim and retrieved passages.

def build_meta_prompt(query: str, passages: list[dict]) -> str:
    ''' 
    Input example: query="Tokyo hosted the 19th G7 Summit"
                   passages=[{"text":"The 19th G7 Summit was held in Tokyo, Japan."}, {...}]
    Output example: A string prompt containing instructions, the question, bullet evidence, and "ANSWER:".
    '''
    context = "\n".join([f"- {p['text']}" for p in passages[:6]]) #Create bullet list from top-6 evidence sentences.

    return (
        "Answer the question using ONLY the evidence bullets.\n" #Do not allow external knowledge.
        "If the answer is not directly stated, output: UNKNOWN.\n" #Force abstention when evidence is missing.
        "Return a short answer (1-5 words).\n\n" #Limit answer length for evaluation.
        f"QUESTION: {query}\n" #Insert the claim/question.
        f"EVIDENCE:\n{context}\n\n" #Insert bullet-point evidence.
        "ANSWER:" #Where the model must write the final answer.
    )
