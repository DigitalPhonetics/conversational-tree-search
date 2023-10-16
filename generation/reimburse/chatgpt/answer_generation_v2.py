# %%
import openai
from tqdm.auto import tqdm
import time
import os
openai.api_key = os.environ["OPENAI_API_KEY"]

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

from data.dataset import ReimburseGraphDataset, StandardGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Question

# %%
reimburse_human_data = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, DataAugmentationLevel.NONE, augmentation_path=None, resource_dir='../../../resources')

# %%
def parse_output(result):
    result_strings = result.get('choices')[0].get("message").get("content").split('\n')
    questions = []
    unnumbered_questions = []
    question_idx = 1
    for question in result_strings:
        question = question.replace('"\n', "").replace('\"', "").strip()
        if question.startswith(f"{question_idx}."):
            questions.append(question.strip(f"{question_idx}.").strip())
        else:
            unnumbered_questions.append(question)
        question_idx += 1
    return questions, unnumbered_questions

# %%
def prompt(node_text: str, answer_text: str, num_paraphrases: int):
    return f"""Generate {num_paraphrases} paraphrases for the response "{answer_text}" to the question {node_text}"""

def api_prompt(prompt: str):
    return [
        {"role": "system", "content": "You are generating semantically similar paraphrases for a given response to some question. The generated response paraphrases should be human-like and short, using frequently used words and phrases only. Present the results in a numbered list."},
        {"role": "user", "content": prompt},
    ]

def api_completion(node_text: str, answer_text: str, num_paraphrases: int):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_prompt(prompt(node_text, answer_text, num_paraphrases))
    )
# %%
from collections import defaultdict
import traceback

NUM_PARAPHRASES = 5

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
                answers, unnumbered_answers = parse_output(response)

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
        

# %% Keyword-based generation

def prompt(node_text: str, answer_text: str, num_paraphrases: int):
    return f"""Generate {num_paraphrases} options for shortening the response "{answer_text}" to the question {node_text}"""

def api_prompt_keywords(prompt: str):
    return [
        {"role": "system", "content": "You are shortening a given response to some question into a keyword-like prompt. Present the results in a numbered list."},
        {"role": "user", "content": prompt},
    ]

def api_completion_keywords(node_text: str, answer_text: str, num_paraphrases: int):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_prompt_keywords(prompt(node_text, answer_text, num_paraphrases))
    )

NUM_KEYWORD_PARAPHRASES = 5


for idx, node in tqdm(enumerate(reimburse_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        done = False
        while not done:
            try:
                response = api_completion_keywords(node.text, answer.text, NUM_KEYWORD_PARAPHRASES)
                answers, unnumbered_answers = parse_output(response)

                generated[answer.key] = generated[answer.key].union(answers)
                generated_unnumbered[answer.key] = generated_unnumbered[answer.key].union(unnumbered_answers)

                num_generated += len(answers)
                num_generated_unnumbered += len(unnumbered_answers)

                if idx % 10 == 0:
                    print(f"Generated: {num_generated}, Unnumbered: {num_generated_unnumbered}")
                
                done = True
            except:
                traceback.print_exc()
                done = True
                print("waiting...")
                time.sleep(15)
        

# %%
import json
with open("../../../resources/en/reimburse/generated/chatgpt/train_answers_v2.json", "w") as f:
    formatted = {}
    for answer_key in generated:
        formatted[answer_key] = list(generated[answer_key])
    json.dump(formatted, f)

with open("../../../resources/en/reimburse/generated/chatgpt/train_answers_unnumbered_v2.json", "w") as f:
    formatted = {}
    for answer_key in generated_unnumbered:
        formatted[answer_key] = list(generated_unnumbered[answer_key])
    json.dump(formatted, f)