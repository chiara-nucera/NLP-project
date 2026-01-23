#Text generation
#Handling text generation using a HuggingFace sequence-to-sequence language model
#This is the part of the project dedicated to give an answer in natural and readable text
#considered a textual prompt => the LLM is used


"""
This file defines a wrapper around a HuggingFace sequence-to-sequence
language model. The generator is used to produce textual answers conditioned on
retrieved and verified evidence. Multiple generations can be sampled to analyze
stability and self-consistency of the model under the same prompt.
"""

from typing import List #to annotate the output type
import torch 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM #HuggingFace utilities


#Wrapper class for HuggingFace text generation models
class HFGenerator:
    def __init__(self, model_name: str, max_new_tokens: int):
        #Loading the tokenizer associated with the model
        self.tok = AutoTokenizer.from_pretrained(model_name)

        #Loading a sequence-to-sequence language model 
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        #Selecting GPU if available, otherwise CPU (on which hardware to execute the calculations of the model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device) #moving the model to the selected device

        #Maximum length of the generated output
        self.max_new_tokens = max_new_tokens


    #Disabling gradient computation since this the LLM is inference-only in this case
    @torch.inference_mode()
    def generate(self, prompt: str, num_samples: int = 1) -> List[str]:
        #num_samples = how many answers the model has to generate for the same prompt
        '''
        Example input: 
        prompt = "Based on the evidence, is the claim 'Paris is the capital of France' true?"
        num_samples = 1

        Example output: 
        [
        "Yes, Paris is the capital of France.",
        "The claim is true: Paris is the capital city of France."
        ]
        '''
        #Tokenizing the input prompt and moving tensors to the correct device
        enc = self.tok(prompt, return_tensors="pt", truncation=True).to(self.device)
        outs = [] #list collecting generated texts

        #Generating multiple answers from the same prompt
        for _ in range(num_samples): #repeating the generation for the same input (prompt)
            gen = self.model.generate( #to generate a sequence of tokens (numbers)
                **enc,
                do_sample=True, #enabling stochastic sampling (for the same 
                                #input the model can output different answers)
                                #Necessary to see if the model is stable or if it changes the answer
                temperature=0.8, #controlling randomness of generation
                max_new_tokens=self.max_new_tokens, #limiting output length
            )

            #Decoding generated token ids into readable text
            text = self.tok.decode(gen[0], skip_special_tokens=True) #transforming the tokens in readable text
            outs.append(text)

        return outs #returning all generated samples
