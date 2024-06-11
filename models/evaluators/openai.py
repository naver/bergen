'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import openai
from tqdm import tqdm
import numpy as np
import os

def openai_api_calculate_cost(usage,model="gpt-4-1106-preview"):
    pricing = {
        'gpt-3.5-turbo': {
            'prompt': 0.0015 ,  #US$1.50 / 1M tokens
            'completion': 0.0020,
        },
        'gpt-4-1106-preview': {
            'prompt': 0.01,
            'completion': 0.03,
        },
        'gpt-4-0125-preview': {
            'prompt': 0.01,
            'completion': 0.03,
        },        
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06,
        },
        'gpt-4o': {
            'prompt': 0.005,  #US$5.00 / 1M tokens
            'completion': 0.015,  #US$15.00 / 1M tokens
        }        
    }

    try:
        model_pricing = pricing[model]
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


def run_llm(client, model_name,messages):
    #model_name="gpt-3.5-turbo"
    #model_name='gpt-4'
    #model_name="gpt-4-0125-preview"
    response = client.chat.completions.create( messages=messages, model=model_name)
    cost = openai_api_calculate_cost(response.usage,model_name)
    #print (cost)
    return response.choices[0].message.content, cost



def create_instruction(question,answer,prediction):
    prefix =  [{'role': 'system',
             'content': "You are an evaluation tool. Just answer by [Yes] or [No]."}]
    prefix.extend([{'role': 'user',
             'content': f"Here is a question, a golden answer and an AI-generated answer. Can you judge whether the AI-generated answer is correct according to the question and golden answer, simply answer Yes or No.\n Question: {question}. \ngolden answer: {answer} \n Generated answer: {prediction}"}
             ]
             )
    return prefix    


# for evaluation
class OpenAI():
    def __init__(self,model):
        self.client = openai.OpenAI(api_key = os.environ.get("OPENAI_API_KEY"),)
        self.model_name=model
    def __call__(self, predictions, references, questions):
        scores=list()
        total_cost=0
        prompt_cost=0
        completion_cost=0
        for q,r,p in (tq:= tqdm(zip(questions,references,predictions),total=len(questions),desc=f"score:  0.0%")):
            prompt = create_instruction(q,r[0],p)
            response,costs = run_llm(self.client,self.model_name,prompt)
            total_cost += costs[0]
            prompt_cost += costs[1]
            completion_cost += costs[2]
            score = 1 if "yes" in response.lower() else 0       
            scores.append(score)
            tq.set_description(f"cost:{total_cost:4.1f} score: {np.mean(scores)* 100:4.1f}%")
        print(total_cost,prompt_cost,completion_cost)
        return np.mean(scores), scores, {"total_cost":total_cost,"prompt_cost":prompt_cost,"completion_cost":completion_cost}
