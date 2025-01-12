# %%
from tqdm.auto import tqdm
import time
import os
import re
from itertools import chain, zip_longest

GPT_VERSION = "gpt-3.5-turbo" # original: gpt-3.5-turbo
SEED = 43
TEMPERATURE = 0.7

from openai import OpenAI

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

with open("../../../openai_api_key.sh", "r") as f:
    line = f.readline()
    env_var_name, api_key = line.split("=")
client = OpenAI(api_key=api_key)

from data.dataset import ReimburseGraphDataset, StandardGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Question

# %%
reimburse_human_data = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, DataAugmentationLevel.NONE, augmentation_path=None, resource_dir='../../../resources')

# %%
def parse_output(result, expected_num: int, mode: str):
    result_strings = []
    initial_splits = result.choices[0].message.content.split('<br>')
    for text in initial_splits:
        splits = text.split("\n")
        result_strings += splits
    
    questions = []
    unnumbered_questions = []
    if len(result_strings) < expected_num:
        unnumbered_questions.append(node)
        print("------", mode, "------")
        print(f"{len(result_strings)} / {expected_num}")
        print(node)
        print("-> results:")
        print(result_strings)
    for question in result_strings:
        cleaned = question.strip().strip("\n").strip()
        cleaned = re.sub(r'^\d+[\.\s]', '', cleaned).strip()
        if len(cleaned) > 0:
            questions.append(cleaned)
    return questions, unnumbered_questions

# %%
def prompt(node_text: str, answer_text: str, num_paraphrases: int):
    return f"""Generate {num_paraphrases} paraphrases for the response "{answer_text}" to the question {node_text}"""

def api_prompt(prompt: str):
    return [
        {"role": "system", "content": "You are generating semantically similar paraphrases for a given response to some question. The generated response paraphrases should be human-like and short, using frequently used words and phrases only. Present the results in a numbered list, separating each paraphrase by a <br> tag."},
        {"role": "user", "content": prompt},
    ]

def api_completion(node_text: str, answer_text: str, num_paraphrases: int):
    return client.chat.completions.create(
        model=GPT_VERSION,
        messages=api_prompt(prompt(node_text, answer_text, num_paraphrases)),
        temperature=TEMPERATURE,
        stream=False,
        seed=SEED
    )
# %%
from collections import defaultdict

NUM_PARAPHRASES = 100

generated_paraphrases = defaultdict(lambda: [])
generated_paraphrases_unnumbered = defaultdict(lambda: [])
num_generated_paraphrases = 0
num_generated_paraphrases_unnumbered = 0

for idx, node in tqdm(enumerate(reimburse_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        done = False
        while not done:
            try:
                response = api_completion(node.text, answer.text, NUM_PARAPHRASES)
                answers, unnumbered_answers = parse_output(response, NUM_PARAPHRASES, 'paraphrase')

                generated_paraphrases[answer.key] += answers
                generated_paraphrases_unnumbered[answer.key] += unnumbered_answers

                num_generated_paraphrases += len(answers)
                num_generated_paraphrases_unnumbered += len(unnumbered_answers)

                if idx % 10 == 0:
                    print(f"Generated: {num_generated_paraphrases}, Unnumbered: {num_generated_paraphrases_unnumbered}")
                
                done = True
            except:
                # traceback.print_exc()
                print("waiting...")
                time.sleep(15)
    #     break
    # break
        

# %% Keyword-based generation

def prompt(node_text: str, answer_text: str, num_paraphrases: int):
    return f"""Generate {num_paraphrases} options for shortening the response "{answer_text}" to the question {node_text}"""

def api_prompt_keywords(prompt: str):
    return [
        {"role": "system", "content": "You are shortening a given response to some question into a keyword-like prompt. Present the results in a numbered list, separating each paraphrase by a <br> tag."},
        {"role": "user", "content": prompt},
    ]

def api_completion_keywords(node_text: str, answer_text: str, num_paraphrases: int):
    return client.chat.completions.create(
        model=GPT_VERSION,
        messages=api_prompt_keywords(prompt(node_text, answer_text, num_paraphrases)),
        temperature=TEMPERATURE,
        stream=False,
        seed=SEED
    )

NUM_KEYWORD_PARAPHRASES = 100

num_generated_shortened = 0
num_generated_shortened_unnumbered = 0
generated_shortened = defaultdict(lambda: [])
generated_shortened_unnumbered = defaultdict(lambda: [])

for idx, node in tqdm(enumerate(reimburse_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        done = False
        while not done:
            try:
                response = api_completion_keywords(node.text, answer.text, NUM_KEYWORD_PARAPHRASES)
                answers, unnumbered_answers = parse_output(response, NUM_KEYWORD_PARAPHRASES, 'shortening')

                generated_shortened[answer.key] += answers
                generated_shortened_unnumbered[answer.key] += unnumbered_answers

                num_generated_shortened += len(answers)
                num_generated_shortened_unnumbered += len(unnumbered_answers)

                if idx % 10 == 0:
                    print(f"Generated: {num_generated_shortened}, Unnumbered: {num_generated_shortened_unnumbered}")
                
                done = True
            except:
                # traceback.print_exc()
                done = True
                print("waiting...")
                time.sleep(15)
    #     break
    # break

# interleave paraphrases + shortened paraphrases for fair data composition in generation studies with different amounts of data
def interleave(l1, l2):
    return [x for x in chain.from_iterable(zip_longest(l1, l2)) if x is not None]

generated = defaultdict(lambda: [])
generated_unnumbered = defaultdict(lambda: [])
for idx, node in tqdm(enumerate(reimburse_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        generated[answer.key] = interleave(generated_paraphrases[answer.key], generated_shortened[answer.key])
        generated_unnumbered[answer.key] = interleave(generated_paraphrases_unnumbered[answer.key], generated_shortened_unnumbered[answer.key])
        

# %%
import json
with open("../../../resources/en/reimburse/generated/chatgpt/thesis/train_answers.json", "w") as f:
    formatted = {}
    for answer_key in generated:
        formatted[answer_key] = generated[answer_key]
    json.dump(formatted, f)

with open("../../../resources/en/reimburse/generated/chatgpt/thesis/train_answers_unnumbered.json", "w") as f:
    formatted = {}
    for answer_key in generated_unnumbered:
        formatted[answer_key] = generated_unnumbered[answer_key]
    json.dump(formatted, f)