# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# %%
import openai
from tqdm.auto import tqdm
import time
openai.api_key = os.environ["OPENAI_API_KEY"]

# %%
import sys
sys.path.append('../../..')
print(os.path.realpath("."))

from data.dataset import ReimburseGraphDataset, StandardGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Question

# %%
reimburse_human_data = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', True, DataAugmentationLevel.NONE, augmentation_path=None, resource_dir='../../../resources')
# %%

def prompt_v3(node_text: str, num_questions: int):
    return f"""Generate {num_questions} questions about the given facts: "{node_text}"""

def prompt_v3_ner(answer_text: str, ner: str, num_questions: int):
    return f"""Generate {num_questions} questions about the entity "{ner}" from the fact: "{answer_text}" """

def api_prompt_v3(prompt: str):
    return [
        {"role": "system", "content": "You are a truthful assistant, generating diverse FAQ-style questions given some facts. The generated questions should be answerable using the given fact only, without additional knowledge. The questions should also be short and human-like. Try to vary the amount of information between questions. Present the results in a numbered list."},
        {"role": "user", "content": prompt},
    ]

def api_completion_v3(node_text: str, num_questions: int):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_prompt_v3(prompt_v3(node_text, num_questions))
    )

def api_completion_v3_ner(answer_text: str, ner: str, num_questions: int):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_prompt_v3(prompt_v3_ner(answer_text, ner, num_questions))
    )


# %%
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,ner', device="cuda:0")

# %%
from statistics import mean

nodes_with_ner = 0
nodes_without_ner = 0
avg_node_ner = []

for node in tqdm(reimburse_human_data.nodes_by_type[NodeType.INFO]):
    context = nlp(node.text)
    if len(context.ents) > 0:
        nodes_with_ner += 1
        avg_node_ner.append(len(context.ents))
    else:
        nodes_without_ner += 1

print("TOTAL INFO NODES", len(reimburse_human_data.nodes_by_type[NodeType.INFO]))
print("NODES WITH NER", nodes_with_ner)
print("NODES WITHOUT NER", nodes_without_ner)
print("AVG NER PER NODE WITH NER", mean(avg_node_ner))

# %%
avg_node_sentence_length = []

for node in reimburse_human_data.nodes_by_type[NodeType.INFO]:
    avg_node_sentence_length.append(node.text.count("."))

print("MAX #SENTENCES PER NODE", max(avg_node_sentence_length))
print("AVG #SENTENCES PER NODE", mean(avg_node_sentence_length))

# %%
from typing import List, Tuple

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
# find a testing candidate
for node in reimburse_human_data.nodes_by_type[NodeType.INFO]:
    results = extract_ner_sentences(node)
    if len(results) > 1:
        print(results)
        break

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
import time
import traceback

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
generated_data_unnumbered = {}

for node in tqdm(reimburse_human_data.nodes_by_type[NodeType.INFO]):
    # use dict indexed by generated text to filter out duplicates
    all_generations = {}
    
    # extract NERs
    named_entities = extract_ner_sentences(node)
    # print(named_entities)

    # Generate questions with NER sentences only, make asking about NER a requirement
    for entity, sentence in named_entities:
        # print("- ENTITY", entity)
        done = False
        while not done:
            try:
                # prompt = prompt_v3_ner(node.text, entity, NUM_QUESTIONS_PER_SENTENCE)
                gen = api_completion_v3_ner(node.text, entity, NUM_QUESTIONS_PER_SENTENCE)
                questions, unnumbered_questions = parse_output(gen)

                for question in questions:
                    key = str(time.time()).replace(".", "")
                    all_generations[key] = {
                        "key": key,
                        "context": "ner",
                        "entity": entity,
                        "dialog_node_key": node.key,
                        "node_text": node.text,
                        "text": question
                    }
                for question in unnumbered_questions:
                    key = str(time.time()).replace(".", "")
                    generated_data_unnumbered[key] = {
                        "key": key,
                        "context": "ner",
                        "entity": entity,
                        "dialog_node_key": node.key,
                        "node_text": node.text,
                        "text": question
                    }
                done = True
            except:
                traceback.print_exc()
                done = True
                print("waiting...")
                time.sleep(15)
    # Generate questions with whole context
    num_node_level_questions = max(NUM_QUESTIONS_PER_SENTENCE, NUM_QUESTIONS - len(named_entities) * NUM_QUESTIONS_PER_SENTENCE)
    # print("NUM GENERIC", num_node_level_questions)
    done = False
    while not done:
        try:
            gen = api_completion_v3(node.text, num_node_level_questions)
            questions, unnumbered_questions = parse_output(gen)

            for question in questions:
                    key = str(time.time()).replace(".", "")
                    all_generations[key] = {
                        "key": key,
                        "context": "node",
                        "dialog_node_key": node.key,
                        "node_text": node.text,
                        "text": question
                    }
            for question in unnumbered_questions:
                key = str(time.time()).replace(".", "")
                generated_data_unnumbered[key] = {
                    "key": key,
                    "context": "node",
                    "dialog_node_key": node.key,
                    "node_text": node.text,
                    "text": question
                }
            done = True
        except:
            traceback.print_exc()
            done = True
            print("waiting...")
            time.sleep(15)
    # filter out duplicates
    uniques = set()
    for question_key in all_generations:
        question = all_generations[question_key]
        if question['text'].lower() in uniques:
            continue # skip duplicate
        else:
            uniques.add(question['text'].lower())
            generated_data[question['key']] = question



# %%
import json
with open("../../../resources/en/reimburse/generated/chatgpt/train_questions_v3.json", "w") as f:
    json.dump(generated_data, f)

with open("../../../resources/en/reimburse/generated/chatgpt/train_questions_v3_unnumbered.json", "w") as f:
    json.dump(generated_data_unnumbered, f)
