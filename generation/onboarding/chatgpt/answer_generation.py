# %%
import openai
from tqdm.auto import tqdm
import os
import time
openai.api_key = os.environ["OPENAI_API_KEY"]

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

from data.dataset import ReimburseGraphDataset, OnboardingGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Question

# %%
onboard_human_data = OnboardingGraphDataset('en/onboarding/train_graph.json', 'en/onboarding/train_answers.json', True, DataAugmentationLevel.NONE, augmentation_path=None, resource_dir='../../../resources')

# %%
def parse_output(result):
    result_strings = result.get('choices')[0].get("message").get("content").split('\n')
    questions = []
    unnumbered_questions = []
    question_idx = 1
    for question in result_strings:
        question = question.strip()
        if question.startswith(f"{question_idx}."):
            questions.append(question.strip(f"{question_idx}.").strip())
        else:
            unnumbered_questions.append(question)
        question_idx += 1
    return questions, unnumbered_questions

# %%
def prompt(node_text: str, answer_text: str, num_paraphrases: int):
    return f"""Generate {num_paraphrases} paraphrases for the answer "{answer_text}" """

def api_prompt(prompt: str):
    return [
        {"role": "system", "content": "You are generating semantically similar paraphrases for a given answer to some question. The generated answer paraphrases should be human-like and short, using frequently used words and phrases only. Present the results in a numbered list."},
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

NUM_PARAPHRASES = 10 

generated = defaultdict(lambda: set())
generated_unnumbered = defaultdict(lambda: set())

num_generated = 0
num_generated_unnumbered = 0

for idx, node in tqdm(enumerate(onboard_human_data.nodes_by_type[NodeType.QUESTION])):
    for answer in node.answers:
        done = False
        while not done:
            try:
                response = api_completion(node.text, answer.text, NUM_PARAPHRASES)
                answers, unnumbered_answers = parse_output(response)

                generated[answer.text.strip().lower()] = generated[answer.text.strip().lower()].union(answers)
                generated_unnumbered[answer.text.strip().lower()] = generated_unnumbered[answer.text.strip().lower()].union(unnumbered_answers)

                num_generated += len(answers)
                num_generated_unnumbered += len(unnumbered_answers)

                if idx % 10 == 0:
                    print(f"Generated: {num_generated}, Unnumbered: {num_generated_unnumbered}")
                
                done = True
            except:
                # traceback.print_exc()
                print("waiting...")
                time.sleep(15)

# %%
import json
with open("../../../resources/en/onboarding/generated/chatgpt/train_answers.json", "w") as f:
    formatted = {}
    for answer_key in generated:
        formatted[answer_key] = list(generated[answer_key])
    json.dump(formatted, f)

with open("../../../resources/en/onboarding/generated/chatgpt/train_answers_unnumbered.json", "w") as f:
    formatted = {}
    for answer_key in generated_unnumbered:
        formatted[answer_key] = list(generated_unnumbered[answer_key])
    json.dump(formatted, f)


