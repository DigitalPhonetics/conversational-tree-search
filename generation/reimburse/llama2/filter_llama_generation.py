# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
DEVICE = 'cuda:0'


# %%
from tqdm.auto import tqdm
from typing import List
import re
import torch

# %%
torch.cuda.device_count()

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/upstage-llama-30b-instruct-2048-GPTQ"
model_basename = "gptq_model-4bit--1g"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=True,
                                          cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/",)

model = AutoGPTQForCausalLM.from_quantized("/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/TheBloke--upstage-llama-30b-instruct-2048-GPTQ",
        # model_basename=model_basename,
        # revision="gptq-4bit-32g-actorder_True",
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# %%
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

def parse_output(prompt: str, output: str) -> List[str]:
    # remove prompt from output first (ends at ### ASSISTANT: )
    return output.replace("<s>", "").strip()[len(prompt):].replace("</s>", "").strip()

# %%
from data.dataset import ReimburseGraphDataset, DataAugmentationLevel

datasets = {
    "train_questions_v1": ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v1.json", resource_dir="../../../resources"),
    "train_questions_v1_ling": ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v1_ling.json", resource_dir="../../../resources"),
    "train_questions_v2": ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v2.json", resource_dir="../../../resources"),
    "train_questions_v2_ling": ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v2_ling.json", resource_dir="../../../resources"),
    "train_questions_v3": ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v3.json", resource_dir="../../../resources"),
    "train_questions_v3_ling": ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v3_ling.json", resource_dir="../../../resources"),
}


# %%
from data.dataset import NodeType, Question
import time
import json

set_seed(42)

system = """You are a truthful assistant deciding if a given question can be answered only using the presented fact, without any additional external knowledge.
Only reply with "yes" or "no"."""

def user(question_text: str, node_text: str) -> str:
    return f'Can the question "{question_text}" be answered without any additional external knowledge and only using the fact: "{node_text}"'

TEMPERATURE = 0.7
MAX_NEW_TOKENS = 1024


for dataset_name in datasets:
    generated_data = {}
    for node in tqdm(datasets[dataset_name].nodes_by_type[NodeType.INFO]):
        for question in node.questions:
            prompt = generate_prompt(system=system, user=user(question.text, node.text)).strip()
            gen = generate_output(prompt=prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
            cleaned = parse_output(prompt, gen)
            generated_data[question.key] = cleaned
    
    with open(f"/mount/arbeitsdaten/asr-2/vaethdk/cts_newcodebase_rollback/conversational-tree-search/resources/en/reimburse/generated/{dataset_name}_with_judgement.json", "w") as f:
        cleaned_data = {}
        for question_key in generated_data:
            question = datasets[dataset_name].questions_by_key[question_key]
            cleaned_data[question_key] = {
                "dialog_node_key": question.parent.key,
                "key": question_key,
                "text": question.text,
                "node_text": question.parent.text,
                "node_type": question.parent.node_type.value,
                "judgement": generated_data[question_key]
            }
        json.dump(cleaned_data, f)

