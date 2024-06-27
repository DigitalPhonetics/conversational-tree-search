# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# %%
from tqdm.auto import tqdm
from typing import List, Tuple
import re
DEVICE = 'cuda:0'
import torch

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

# %%
import time

# %%
a = torch.zeros(1,1,device=DEVICE)

# %%
# !GITHUB_ACTIONS=true pip install auto-gptq

# %%
from data.dataset import StandardGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Question
human_data_train = StandardGraphDataset('en/diagnose/train_graph.json', 'en/diagnose/train_graph.json', False, augmentation=DataAugmentationLevel.NONE, resource_dir="../../../resources")

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# %%
model_name_or_path = "TheBloke/upstage-llama-30b-instruct-2048-GPTQ"
model_basename = "gptq_model-4bit--1g"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=True,
                                          cache_dir="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/")


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
# ## Generate Question Synonyms

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
# check that we don't have any answer synonyms
for node in human_data_train.node_list:
    for question in node.questions:
        human_data_train.question_list.remove(question)
        del human_data_train.questions_by_key[question.key]
    node.questions.clear()
assert len(human_data_train.question_list) == 0
assert len(human_data_train.questions_by_key) == 0

# %%
def parse_output(original_question: str, prompt: str, output: str, num_paraphrases: int) -> List[str]:
    # remove prompt from output first (ends at ### ASSISTANT: )
    questions = []
    cleaned = output[len(prompt):]
    
    if not "1." in cleaned: 
        print("NO LIST FOR QUESTION", original_question)
        return questions
    
    for i in range(1, num_paraphrases+1):
        if not f"{i}." in cleaned: 
            print(f" - NO {i}. CANDIDATE FOR QUESTION", original_question)
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
        questions.append(cleaned[start_idx:end_idx].replace("</s>", "").strip())

        cleaned = cleaned[end_idx:] # remove i. line
    return questions

# %% [markdown]
# # DATA GENERATION V3: SHORTER QUESTIONS + SPLIT NODE CONTEXT

# %% [markdown]
# 1. Try to generate shorter questions (-> change prompt)
# 2. Try to generate more diverse questions
#     1. Detect relevant sentences in node text via NER tool (also detects time, quantities, ...)
#     2. Generate questions for whole node context, then for only relevant sentences / sub-sentences of node
#     3. Choose amount of questions to be generated depending on amount of extracted NERs?

# %%
import stanza

# %%
# stanza.download(lang="en", model_dir=".models/")

# %%
nlp = stanza.Pipeline('en', processors='tokenize,ner', device="cuda:0")

# %%
from statistics import mean

nodes_with_ner = 0
nodes_without_ner = 0
avg_node_ner = []

for node in tqdm(human_data_train.nodes_by_type[NodeType.INFO]):
    context = nlp(node.text)
    if len(context.ents) > 0:
        nodes_with_ner += 1
        avg_node_ner.append(len(context.ents))
    else:
        nodes_without_ner += 1

print("TOTAL INFO NODES", len(human_data_train.nodes_by_type[NodeType.INFO]))
print("NODES WITH NER", nodes_with_ner)
print("NODES WITHOUT NER", nodes_without_ner)
print("AVG NER PER NODE WITH NER", mean(avg_node_ner))

# %%
avg_node_sentence_length = []

for node in human_data_train.nodes_by_type[NodeType.INFO]:
    avg_node_sentence_length.append(node.text.count("."))

print("MAX #SENTENCES PER NODE", max(avg_node_sentence_length))
print("AVG #SENTENCES PER NODE", mean(avg_node_sentence_length))

# %%
def extract_ner_sentences(node: DialogNode) -> List[Tuple[str, str]]:
    """
    Extract all sentences from node text that mention NER's.
    Returns them as a list of tuples, where each tuple contains
        1. the name of the entity
        2. the sentence containing that entity
    """
    results = []
    context = nlp(node.text)
    entities = context.ents
    for entity in entities:
        start_idx = entity.start_char
        end_idx = entity.end_char
        # expand start index to beginning of sentence
        while start_idx > 0 and node.text[start_idx-1] != ".":
            start_idx -= 1
        # expand end index to end of sentence
        while end_idx < len(node.text) and node.text[end_idx-1] != ".":
            end_idx += 1
        results.append((entity.text, node.text[start_idx:end_idx]))
    return results

# %%
system = """You are a helpful assistant creating a list of diverse FAQ-style questions from given facts.
Only generate questions that can be answered by the given facts, without any external knowledge.
Use casual language.
Prefer short questions.
Order the generated paraphrases in a numbered list."""

def user(answer_text: str, num_paraphrases: int) -> str:
    return f'Generate {num_paraphrases} short and diverse FAQ-style questions from the fact: "{answer_text}"'

def user_ner(answer_text: str, ner: str, num_paraphrases: int) -> str:
    return f'Generate {num_paraphrases} short and diverse FAQ-style questions about the entity "{ner}" from the fact: "{answer_text}"'


NUM_QUESTIONS = 10
NUM_QUESTIONS_PER_SENTENCE = 3
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 1024
generated_data = {}

set_seed(42)

for node in tqdm(human_data_train.nodes_by_type[NodeType.INFO]):
    # use dict indexed by generated text to filter out duplicates
    all_generations = {}
    uniques = set()
    
    # extract NERs
    named_entities = extract_ner_sentences(node)

    # Generate questions with NER sentences only, make asking about NER a requirement
    for entity, sentence in named_entities:
        prompt = generate_prompt(system=system, user=user_ner(node.text, entity, NUM_QUESTIONS_PER_SENTENCE))
        gen = generate_output(prompt=prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        candidates = parse_output(original_question=node.text, prompt=prompt, output=gen, num_paraphrases=NUM_QUESTIONS_PER_SENTENCE)
        for candidate_idx, candidate in enumerate(candidates):
            key = str(time.time()).replace(".", "")
            cleaned_candidate = candidate.replace(f"{candidate_idx+1}.", "").strip()
            all_generations[cleaned_candidate] = {
                "context": "ner",
                "entity": entity,
                "dialog_node_key": node.key,
                "key": key,
                "text": cleaned_candidate
            }

    # Generate questions with whole context
    num_node_level_questions = max(NUM_QUESTIONS_PER_SENTENCE, NUM_QUESTIONS - len(named_entities) * NUM_QUESTIONS_PER_SENTENCE)
    prompt = generate_prompt(system=system, user=user(node.text, num_node_level_questions))
    gen = generate_output(prompt=prompt, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
    candidates = parse_output(original_question=node.text, prompt=prompt, output=gen, num_paraphrases=NUM_QUESTIONS_PER_SENTENCE)
    for candidate_idx, candidate in enumerate(candidates):
        key = str(time.time()).replace(".", "")
        cleaned_candidate = candidate.replace(f"{candidate_idx+1}.", "").strip()
        all_generations[cleaned_candidate] = {
            "context": "node",
            "dialog_node_key": node.key,
            "key": key,
            "text": cleaned_candidate
        }
    
    # add filtered questions to generated dataset
    for text in all_generations:
        entry = all_generations[text]
        generated_data[entry["key"]] = entry


# %%
import json

cleaned_data = {}
for key in generated_data:
    node = human_data_train.nodes_by_key[generated_data[key]['dialog_node_key']]
    cleaned_data[key] = generated_data[key]
    for i in range (1, NUM_QUESTIONS+1):
        cleaned_data[key]['text'] = cleaned_data[key]['text'].replace(f"{i}.", "").strip()
    cleaned_data[key]["node_text"] = node.text
    cleaned_data[key]["node_type"] = node.node_type.value

with open("../../../resources/en/diagnose/generated/train_questions_v3.json", "w") as f:
    json.dump(cleaned_data, f)

# %%



