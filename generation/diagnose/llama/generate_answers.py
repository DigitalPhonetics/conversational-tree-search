# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# %%
from tqdm.auto import tqdm
from typing import List
import re

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

# %%
DEVICE = 'cuda:0'
import torch
a= torch.zeros(1,1,device=DEVICE)

# %%
torch.cuda.device_count()

# %%
# !GITHUB_ACTIONS=true pip install auto-gptq

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# %%
model_name_or_path = "TheBloke/upstage-llama-30b-instruct-2048-GPTQ"
model_basename = "gptq_model-4bit--1g"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=True,
                                          cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/",)


# %% [markdown]
# 

# %%
model = AutoGPTQForCausalLM.from_quantized("/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/TheBloke--upstage-llama-30b-instruct-2048-GPTQ",
        # model_basename=model_basename,
        # revision="gptq-4bit-32g-actorder_True",
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# %% [markdown]
# ### System:
# {System}
# 
# ### User:
# {User}
# 
# ### Assistant:
# {Assistant}

# %% [markdown]
# ## Generate Answer Synonyms

# %%
from data.dataset import StandardGraphDataset
import torch

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


# %%
from data.dataset import DataAugmentationLevel

train = StandardGraphDataset('en/diagnose/train_graph.json', "en/diagnose/train_graph.json", False, augmentation=DataAugmentationLevel.NONE, augmentation_path=None, resource_dir="../../../resources")

# %%
# check that we don't have any answer synonyms
for answer_candidate in train.answer_synonyms:
    assert len(train.answer_synonyms[answer_candidate]) == 1

# %%
def parse_output(original_answer: str, prompt: str, output: str, num_paraphrases: int) -> List[str]:
    # remove prompt from output first (ends at ### ASSISTANT: )
    answers = []
    cleaned = output[len(prompt):]
    
    if not "1." in cleaned: 
        print("NO LIST FOR ANSWER", original_answer)
        return answers
    
    for i in range(1, num_paraphrases+1):
        if not f"{i}." in cleaned: 
            print(f" - NO {i}. CANDIDATE FOR ANSWER", original_answer)
            continue

        start_idx = cleaned.find(f"{i}.") # find i. line
        end_idx = cleaned.find("\n", start_idx) # read until line end 
        if i == num_paraphrases and end_idx == -1:
            # last line might not have line break
            end_idx = len(cleaned)
        if start_idx == -1 or end_idx == -1:
            print(f" - INDEX PROBLEM FOR {i}. CANDIDATE: ({start_idx}, {end_idx})")
            continue
        # parse answer
        answers.append(cleaned[start_idx:end_idx].replace(f"{i}.", "").replace("</s>", "").strip())

        cleaned = cleaned[end_idx:] # remove i. line
    return answers

# %%
system = """You are a helpful assistant creating a list of paraphrases for a given answer to a question.
Order the generated paraphrases in a numbered list."""
def user(answer_text: str, num_paraphrases: int) -> str:
    return f'Generate {num_paraphrases} paraphrases for the answer "{answer_text}"'

NUM_PARAPHRASES = 10
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 512
generated_data = {}
for answer_key in tqdm(train.answer_synonyms):
    prompt = generate_prompt(system=system, user=user(answer_key, NUM_PARAPHRASES))
    gen = generate_output(prompt=prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
    candidates = parse_output(original_answer=answer_key, prompt=prompt, output=gen, num_paraphrases=NUM_PARAPHRASES)
    generated_data[answer_key] = candidates
    

# %%
import json

with open("../../../resources/en/diagnose/generated/train_answers.json", "w") as f:
    json.dump(generated_data, f)

# %%



