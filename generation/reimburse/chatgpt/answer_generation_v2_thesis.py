# %%
from tqdm.auto import tqdm
import time
import os
import re


GPT_VERSION = "gpt-4o-mini" # original: gpt-3.5-turbo
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
    result_strings = result.choices[0].message.content.split('<br>')
    questions = []
    unnumbered_questions = []
    if len(result_strings) != expected_num:
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
import traceback

NUM_PARAPHRASES = 50

generated = defaultdict(lambda: set())
generated_unnumbered = defaultdict(lambda: set())

num_generated = 0
num_generated_unnumbered = 0

for idx, node in tqdm(enumerate(reimburse_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        done = False
        while not done:
            try:
                response = api_completion(node.text, answer.text, NUM_PARAPHRASES)
                answers, unnumbered_answers = parse_output(response, NUM_PARAPHRASES, 'paraphrase')

                generated[answer.key] = generated[answer.key].union(answers)
                generated_unnumbered[answer.key] = generated_unnumbered[answer.key].union(unnumbered_answers)

                num_generated += len(answers)
                num_generated_unnumbered += len(unnumbered_answers)

                if idx % 10 == 0:
                    print(f"Generated: {num_generated}, Unnumbered: {num_generated_unnumbered}")
                
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

NUM_KEYWORD_PARAPHRASES = 25

for idx, node in tqdm(enumerate(reimburse_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        done = False
        while not done:
            try:
                response = api_completion_keywords(node.text, answer.text, NUM_KEYWORD_PARAPHRASES)
                answers, unnumbered_answers = parse_output(response, NUM_KEYWORD_PARAPHRASES, 'shortening')

                generated[answer.key] = generated[answer.key].union(answers)
                generated_unnumbered[answer.key] = generated_unnumbered[answer.key].union(unnumbered_answers)

                num_generated += len(answers)
                num_generated_unnumbered += len(unnumbered_answers)

                if idx % 10 == 0:
                    print(f"Generated: {num_generated}, Unnumbered: {num_generated_unnumbered}")
                
                done = True
            except:
                # traceback.print_exc()
                done = True
                print("waiting...")
                time.sleep(15)
        # break
    # break
        

# %%
import json
with open("../../../resources/en/reimburse/generated/chatgpt/thesis/train_answers_v2.json", "w") as f:
    formatted = {}
    for answer_key in generated:
        formatted[answer_key] = list(generated[answer_key])
    json.dump(formatted, f)

with open("../../../resources/en/reimburse/generated/chatgpt/thesis/train_answers_unnumbered_v2.json", "w") as f:
    formatted = {}
    for answer_key in generated_unnumbered:
        formatted[answer_key] = list(generated_unnumbered[answer_key])
    json.dump(formatted, f)