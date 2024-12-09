'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import openai
from tqdm import tqdm
import numpy as np
import os
import random


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



def create_instruction(question: str, answer: str, prediction: str):
    prefix =  [{'role': 'system',
             'content': "You are an evaluation tool. Just answer by {Yes} or {No}."}]
    prefix.extend([{'role': 'user',
             'content': f"Here is a question, a golden answer and an AI-generated answer. Can you judge whether the AI-generated answer is correct according to the question and golden answer, simply answer {{Yes}} or {{No}}.\n Question: {question}. \ngolden answer: {answer} \n Generated answer: {prediction}.\Response:"}
             ]
             )
    return prefix    


def create_pairwise_instruction(question, ref_answer, answer_1, answer_2):
    prefix =  [{
        'role': 'system',
        'content': "You are a helpful assistant, that ranks models by the quality of their answers. Please act as an impartial judge. Do not allow the length of the responses to influence your evaluation. Be as objective as possible."
        }]
    prefix.extend([{
        'role': 'user',
        'content' :  f"Here is a question, a ground truth answer, an AI-generated answer 1 and an AI-generated answer 2. Which answer is the most correct one ? Simply answer {{1}} if the first is better, {{2}} if the second is better and {{3}} if it's a tie. \n Question: {question}.\n Ground truth answer: {ref_answer}.\n Answer 1: {answer_1}.\n Answer 2: {answer_2}."
        }])
    return prefix    

# for evaluation
class OpenAI():
    
    def __init__(self, model):
        self.client = openai.OpenAI(api_key = os.environ.get("OPENAI_API_KEY"),)
        self.model_name=model
        
    def __call__(self, predictions, references, questions):
        scores = list()
        weird = list()
        total_cost = 0
        prompt_cost = 0
        completion_cost = 0
        for q,r,p in (tq:= tqdm(zip(questions,references,predictions),total=len(questions),desc="score:  0.0%")):
            prompt = create_instruction(q,r[0],p)
            response, costs = run_llm(self.client,self.model_name,prompt)
            total_cost += costs[0]
            prompt_cost += costs[1]
            completion_cost += costs[2]
            score = 1 if "yes" in response.lower() else 0       
            scores.append(score)
            weird.extend([ 1 if ("no" not in response.lower() and "yes" not in response.lower()) else 0 ])
            tq.set_description(f"cost:{total_cost:4.1f} score: {np.mean(scores)* 100:4.1f}% weird {np.mean(weird)* 100:4.1f}%")
        print(total_cost,prompt_cost,completion_cost)
        return np.mean(scores), scores, {"total_cost":total_cost,"prompt_cost":prompt_cost,"completion_cost":completion_cost}
    
    def pairwise_win_rate(self, predictions, opponent_predictions, references, questions):
        assert len(predictions) == len(opponent_predictions)
        scores = []
        weird = []
        total_cost = 0
        prompt_cost = 0
        completion_cost = 0
        for pred_1, pred_2, ref_answer, question in (tq:= tqdm(zip(predictions, opponent_predictions, references, questions), total=len(questions),desc="score:  0.0%")):
            
            # Randomly switch order to prevent position bias in judge
            switch_order = (random.randint(0, 1) == 1)
            if switch_order:
                pred_1, pred_2 = pred_2, pred_1
                
            prompt = create_pairwise_instruction(question, ref_answer[0], answer_1=pred_1, answer_2=pred_2)
            response, costs = run_llm(self.client,self.model_name,prompt)
            total_cost += costs[0]
            prompt_cost += costs[1]
            completion_cost += costs[2]
            score = None
            if '1' in response.lower():
                score = 1
                w = 0
            elif '2' in response.lower():
                score = 0
                w = 0
            elif '3' in response.lower():
                score = 0.5
                w = 0
            else:
                score = 0.5 # tie by default
                w = 1
            
            if switch_order:
                score = 1 - score
                
            scores.append(score)
            weird.append(w)
            tq.set_description(f"cost:{total_cost:4.1f} win: {scores.count(1)*100./len(scores):4.1f}% tie {scores.count(0.5)*100./len(scores):4.1f}% lose {scores.count(0)*100./len(scores):4.1f}%  weird {np.mean(weird)* 100:4.1f}%")
        print(total_cost, prompt_cost, completion_cost)
        avg_scores = {
            'win': scores.count(1)*100./len(scores),
            'tie': scores.count(0.5)*100./len(scores),
            'lose': scores.count(0)*100./len(scores)
        }
        return avg_scores, scores, {"total_cost":total_cost,"prompt_cost":prompt_cost,"completion_cost":completion_cost}
            
        