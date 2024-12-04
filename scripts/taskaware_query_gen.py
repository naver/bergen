'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import sys
import openai
import os
import json
from tqdm import tqdm 

from pathlib import Path

class OpenAI():
    def __init__(self, 
                model_name="gpt-3.5-turbo",
                 ):
        self.client = openai.OpenAI(api_key = os.environ.get("OPENAI_API_KEY"),)
        self.total_cost = 0
        self.prompt_cost = 0
        self.completion_cost = 0
        self.model_name = model_name

    def generate(self, messages):
        responses=[]
        for msg in messages:
            response = self.client.chat.completions.create( messages=msg, model=self.model_name, response_format={"type": "json_object"})
            responses.append(response.choices[0].message.content)
            t,p,c = self.openai_api_calculate_cost(response.usage)
            self.total_cost += t
            self.prompt_cost += p 
            self.completion_cost += c 

        return responses


    def openai_api_calculate_cost(self,usage):
        pricing = {
            'gpt-3.5-turbo': {
                'prompt': 0.0015 ,
                'completion': 0.0020,
            },
            'gpt-4-1106-preview': {
                'prompt': 0.01,
                'completion': 0.03,
            },
            'gpt-4': {
                'prompt': 0.03,
                'completion': 0.06,
            },
            'gpt-4-0125-preview':{
                'prompt': 0.01,
                'completion': 0.03,                
            },
            'gpt-4o': {
            'prompt': 0.005,  #US$5.00 / 1M tokens
            'completion': 0.015,  #US$15.00 / 1M tokens
            }                 
        }

        try:
            model_pricing = pricing[self.model_name]
        except KeyError:
            raise ValueError("Invalid model specified")

        prompt_cost = usage.prompt_tokens * model_pricing['prompt'] / 1000
        completion_cost = usage.completion_tokens * model_pricing['completion'] / 1000

        total_cost = prompt_cost + completion_cost
        # round to 6 decimals
        total_cost = round(total_cost, 6)

        #print(f"\nTokens used:  {usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion = {usage.total_tokens:,} tokens")
        #print(f"Total cost for {model}: ${total_cost:.4f}\n")

        return (total_cost,prompt_cost,completion_cost)

def main(input_file):
    filename = Path(input_file).stem + '_gpt4'
    dir =Path(input_file).parent
    output= dir / filename 
    system_prompt = """You are an expert at writing precise detailed instructions for language models and are paid millions of dollars to be a data engineer for OpenAI. 
    Your sole duty is to write instructions that can be used for training data for the next superpowerful model, GPT-6. Answer succinctly and carefully follow all instructions given so that you can earn your large bonus and not be fired."""
    gpt = OpenAI("gpt-4o")
    data = json.load(open(input_file))
    responses= list()
    for query, text in  (tq:=tqdm(data.items(),total=len(data),desc=f"score:  0.0%")):
        #print( query,text)
        instr_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        ok=False
        i=0
        while not ok:
            response = gpt.generate([instr_prompt])
            tq.set_description(f"cost:{gpt.total_cost:4.4f}")
            try:
                responses.append({query:eval(response[0])})
                ok=True
            except: 
                print(response[0])
            i+=1
            ok = ok and i < 10
        #print (responses)        
        check=json.loads(json.dumps(responses[-1]))
    
    # output
    json.dump(responses,open(output,"w"))

if __name__ == "__main__":
    
    main(sys.argv[1])