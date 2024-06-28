import json
from tqdm.auto import tqdm
import time
import openai
import os
openai.api_key = os.environ["OPENAI_API_KEY"]

def prompt(node_text: str, question: str):
    return f"""Can the question "{question}" be answered using only the following facts: "{node_text}"? Answer with yes or no."""

def api_prompt(prompt: str):
    return [
        {"role": "system", "content": "You are a truthful assistant, judging if a question can be answered by some given facts. To be marked as answereable, the question should be answerable by the given facts only and not require any additional resources. You only reply with yes or no."},
        {"role": "user", "content": prompt},
    ]

def api_completion(node_text: str, question: str):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_prompt(prompt(node_text, question))
    )

with open("../../../resources/en/reimburse/generated/train_questions_v2_ling.json", "r") as f:
    data = json.load(f)

print(len(data), "samples")

filtered_questions = []
keys = set()

for sample_idx, sample_key in tqdm(enumerate(data)):
    done = False
    while not done:
        if sample_key in keys:
            done = True
        else:
            try:
                sample = data[sample_key]
                dialog_node_key = sample['dialog_node_key']
                question = sample['text']
                node_text = sample['node_text']
                completion = api_completion(node_text, question)
                judgement = completion.get("choices")[0].get("message")["content"]
                sample['judgement'] = judgement
                filtered_questions.append(sample)
                keys.add(sample_key)
                done = True
                # assert len(filtered_questions) == sample_idx + 1
            except:
                # traceback.print_exc()
                print("Waiting")
                time.sleep(30)

with open("../../../resources/en/reimburse/generated/train_questions_v2_ling_filtered_chatgpt.json", "w") as f:
    json.dump(filtered_questions, f)
