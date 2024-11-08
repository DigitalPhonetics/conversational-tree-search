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

from data.dataset import StandardGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Question

# %%
human_data = StandardGraphDataset('en/onboarding/train_graph.json', 'en/onboarding/train_answers.json', True, DataAugmentationLevel.NONE, augmentation_path=None, resource_dir='../../../resources')


# %%
def prompt(node_text: str, num_questions: int):
    return f"""Generate {num_questions} questions about the given facts: "{node_text}"""

def api_prompt(prompt: str):
    return [
        {"role": "system", "content": "You are a truthful assistant, generating diverse FAQ-style questions given some facts. The generated questions should be answerable using the given fact only, without additional knowledge. The questions should also be short and human-like. Try to vary the amount of information between questions. Present the results in a numbered list."},
        {"role": "user", "content": prompt},
    ]

def api_completion(node_text: str, num_questions: int):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_prompt(prompt(node_text, num_questions))
    )

# %%
# NUM_QUESTIONS = 10 

# for node in tqdm(human_data.nodes_by_type[NodeType.INFO]):
#     result = api_completion(node.text, NUM_QUESTIONS)
#     break


# %%
# print(result)

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
# good, bad = parse_output(result)
# print(good)

# %%
import traceback

NUM_QUESTIONS = 10 

generated = {}
generated_unnumbered = {}

num_generated = 0
num_generated_unnumbered = 0
# %%
for idx, node in tqdm(enumerate(human_data.nodes_by_type[NodeType.INFO])):
    done = False
    while not done:
        try:
            response = api_completion(node.text, NUM_QUESTIONS)
            questions, unnumbered_questions = parse_output(response)

            generated[node.key] = questions
            generated_unnumbered[node.key] = unnumbered_questions

            num_generated += len(questions)
            num_generated_unnumbered += len(unnumbered_questions)

            if idx % 10 == 0:
                print(f"Generated: {num_generated}, Unnumbered: {num_generated_unnumbered}")
            
            done = True
        except:
            # traceback.print_exc()
            print("waiting...")
            time.sleep(15)

# %%
import json
with open("../../../resources/en/onboarding/generated/chatgpt/train_questions_v2.json", "w") as f:
    json.dump(generated, f)

with open("../../../resources/en/onboarding/generated/chatgpt/train_questions_v2_unnumbered.json", "w") as f:
    json.dump(generated_unnumbered, f)

# %%



