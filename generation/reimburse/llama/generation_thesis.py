import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
DEVICE = 'cuda:0'

import re
import sys
sys.path.append('../../..')
print(os.path.realpath("."))


from tqdm.auto import tqdm
from typing import List
import torch
from data.dataset import NodeType
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from data.dataset import ReimburseGraphDataset, DataAugmentationLevel, NodeType


model_name_or_path = "TheBloke/upstage-llama-30b-instruct-2048-GPTQ"
model_basename = "gptq_model-4bit--1g"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main",
                                             cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=True,
                                          cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/")


human_data_train = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', False, augmentation=DataAugmentationLevel.NONE, resource_dir="../../../resources/")

def generate_prompt(system: str, user: str) -> str:
    return f"""
    ### System:
    {system}

    ### User:
    {user}

    ### Assistant:"""

def generate_output(prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> torch.FloatTensor:
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=temperature, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0])

def parse_output(original_question: str, prompt: str, output: str, num_paraphrases: int) -> List[str]:
    # remove prompt from output first (ends at ### ASSISTANT: )
    questions = []
    cleaned = output[len(prompt):].split("\n")
    if len(cleaned) != num_paraphrases:
        print(f"- PROBLEM: Generated {len(cleaned)} questions for node: {original_question}")

    for question in cleaned:
        cleaned = question.strip().strip("\n").strip()
        cleaned = re.sub(r'^\d+[\.\s]', '', cleaned).strip()
        if len(cleaned) > 0:
            questions.append(cleaned.replace("</s>", "").strip())
        
        
        # if not "1." in cleaned: 
        #     print("NO LIST FOR QUESTION", original_question)
        #     return questions
        
        # for i in range(1, num_paraphrases+1):
        #     if not f"{i}." in cleaned: 
        #         print(f" - NO {i}. CANDIDATE FOR QUESTION", original_question)
        #         continue

            # start_idx = cleaned.find(f"{i}.") # find i. line
            # end_idx = cleaned.find("\n", start_idx) # read until line end 
            # if i == num_paraphrases and end_idx == -1:
            #     # last line might not have line break
            #     end_idx = len(cleaned)
            # if start_idx == -1 or end_idx == -1:
            #     print(f" - INDEX PROBLEM FOR {i}. CANDIDATE: ({start_idx}, {end_idx})")
            #     continue
            # # parse answer
            # questions.append(cleaned[start_idx:end_idx].replace("</s>", "").strip())

            # cleaned = cleaned[end_idx:] # remove i. line
    return questions



set_seed(42)

system = """You are a helpful assistant creating a list of diverse FAQ-style questions from given facts.
Only generate questions that can be answered by the given facts, without any external knowledge.
Use casual language.
Prefer short questions.
Order the generated questions in a numbered list, with a new line per question."""

def user(answer_text: str, num_paraphrases: int) -> str:
    return f'Generate {num_paraphrases} short and diverse FAQ-style questions from the fact: "{answer_text}"'

NUM_QUESTIONS = 200
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 24000
generated_data = {}

for node in tqdm(human_data_train.nodes_by_type[NodeType.INFO]):
    prompt = generate_prompt(system=system, user=user(node.text, NUM_QUESTIONS))
    gen = generate_output(prompt=prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
    candidates = parse_output(original_question=node.text, prompt=prompt, output=gen, num_paraphrases=NUM_QUESTIONS)
    for candidate in candidates:
        key = str(time.time()).replace(".", "")
        generated_data[key] = {
            "dialog_node_key": node.key,
            "key": key,
            "text": candidate,
        }


cleaned_data = {}
for key in generated_data:
    node = human_data_train.nodes_by_key[generated_data[key]['dialog_node_key']]
    cleaned_data[key] = generated_data[key]
    for i in range (1, NUM_QUESTIONS+1):
        cleaned_data[key]['text'] = cleaned_data[key]['text'].replace(f"{i}.", "").strip()
    cleaned_data[key]["node_text"] = node.text
    cleaned_data[key]["node_type"] = node.node_type.value

with open("train_questions_200_llama.json", "w") as f:
    json.dump(cleaned_data, f)
