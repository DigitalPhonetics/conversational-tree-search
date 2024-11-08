from copy import deepcopy
import itertools
import os
import time
import traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
os.environ['TRANSFORMERS_CACHE'] = '/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/llm'

from sentence_transformers import SentenceTransformer
import torch 
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, set_seed

from data.dataset import ReimburseGraphDataset, DataAugmentationLevel, NodeType, DialogNode, NodeType, GraphDataset, StandardGraphDataset
import json
from typing import List, Set, Tuple
from statistics import mean
from tqdm import tqdm

from openai import OpenAI
# from pynvml import *
 

USE_PREFILTER = False
USE_INCONTEXT_EXAMPLES = True
USE_JUSTIFICATIONS = True

NUM_EPISODES = 500
TOP_K = 15
SEED = 43
TEMPERATURE = 0.0
MODEL = "gemma2"
DATA = "reimburse" # reimburse # diagnose # onboarding
MODE = "train" # train
PROMPT = 1 # 1 # 2
GUIDED_FREE_RATIO = 0.5
NODE_TYPES = [NodeType.INFO, NodeType.QUESTION]
STRICT = False
 

def get_gpu_utilization() -> int:
    # returns gpu memory usage in MB
    return torch.cuda.memory_allocated()


mem_initial_consumption = get_gpu_utilization()
print("INITIAL MEMORY", mem_initial_consumption)



if MODEL == "llama3":
    pipe = pipeline(
        "text-generation",
        model= "meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.float16},
        device='cuda:0'
    )
elif MODEL == "gemma2":
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-9b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda:0",
    )
elif "gpt" in MODEL:
    pipe = None
elif MODEL == "phi":
    model_id = "microsoft/Phi-3-medium-4k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0", 
        torch_dtype=torch.float16, 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + '<|end|>' }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + '<|end|>' }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

# Mono-Lingual
bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1",
                                device="cuda:0", 
                                cache_folder="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/")


mem_model_consumption = get_gpu_utilization() - mem_initial_consumption

print("MODEL MEM CONSUMPTION", mem_model_consumption)
mem_baseline = get_gpu_utilization()




test_data = ReimburseGraphDataset(graph_path='en/reimburse/test_graph.json', answer_path='en/reimburse/test_answers.json', 
                                  use_answer_synonyms=True,
                                  augmentation=DataAugmentationLevel.NONE, augmentation_path=None,
                                  resource_dir="./resources/",
                                  question_limit=0, answer_limit=0, language="en")

with open("openai_api_key.sh", "r") as f:
    line = f.readline()
    env_var_name, api_key = line.split("=")

client = OpenAI(api_key=api_key)


# %%
def format_prompts_gemma2(messages: List[dict]) -> List[dict]:
    results = []
    # gemma 2 does not support system role, replace by user role
    for msg in messages:
        if msg['role'] == "system":
            msg['role'] = 'user'
    # gemma 2 does not support 2 consecutive user messages, merge them together
    last_role = ""
    for msg in messages:
        if not last_role == msg['role']:
            last_role = msg['role']
            results.append(msg)
        else:
            results[-1]['content'] += f"""=====
            {msg['content']}
            """
    return results


# %%
def generate_output(model: str, messages, temperature: float=0.7, seed: int = None, max_new_tokens: int = 50000, batch_size: int = 1) -> str:
    if model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
        if not isinstance(seed, type(None)):
            kwargs = {"seed": seed}
        else:
            kwargs = {}
        output = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
            **kwargs
        )
        raw = output.choices[0].message.content
    elif model in ["llama3", "gemma2", "phi"]:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
        if temperature > 0.0:
            kwargs["temperature"] = temperature
            kwargs["do_sample"] = True
        elif temperature == 0:
            kwargs["do_sample"] = False
        if model == "gemma2":
            messages = format_prompts_gemma2(messages)
            
        outputs = pipe(
            messages,
            pad_token_id=pipe.tokenizer.eos_token_id,
            batch_size=batch_size,
            **kwargs,
        )
        raw = outputs[0]["generated_text"][-1]['content']
    return raw



def get_node_candidate_list_by_type(data: GraphDataset, node_types: List = [NodeType.INFO]) -> List[DialogNode]:
    nodes_by_type = itertools.chain.from_iterable([data.nodes_by_type[node_type] for node_type in node_types])
    nodes_with_questions = filter(lambda node: len(node.questions) > 0, nodes_by_type)
    return list(nodes_with_questions)

def calculate_node_text_embeddings(bi_encoder: SentenceTransformer, node_list: List[DialogNode]):
    docs = []
    for node in node_list:
        docs.append(node.text)
    return bi_encoder.encode(docs, convert_to_tensor=True, batch_size=512)

def calculate_similarity(bi_encoder: SentenceTransformer, query: str, k: int = 5, info_node_embeddings=None):
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True, batch_size=512)

    similarity_scores = bi_encoder.similarity(query_embedding, info_node_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=k)
    return scores, indices

# %%


def get_goal_candidates_fullcontext(node_list: DialogNode, query: str, model: str = "llama3", seed: int = 43, temperature: float = 0.0, verbose: bool =False, strict: bool = True, system_prompt: str = "") -> Tuple[Set[int], str]:
    # step 1: use LLM to determine which candidates are useful
    facts = []
    for idx, node in enumerate(node_list):
        facts.append({
            "id": idx,
            "fact": node.text
        })
    assert len(node_list) == 76
        # print(f"-fact {idx}: {scores[idx]} - {facts[-1]['fact']}")

    user = f"""======= Facts =======
    {json.dumps(facts)}

    ======= Query =======
    "{query}"
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user}
    ]

    outputs = generate_output(model=model, messages=messages, temperature=temperature, seed=seed)
    raw_output = deepcopy(outputs)
    outputs = outputs.replace("\n", "").strip("(").strip(")").strip("'").strip()
    try:
        results = json.loads(outputs)
        candidate_ids = set()
        
        relevant = 0
        if strict:
            assert len(results) == len(facts)
        for idx, res in enumerate(results):
            res_idx = int(res['id'])
            if idx != res_idx:
                if strict:
                    assert idx == res_idx, "QUERY: {query}, outputs: {outputs}"
                # else:
                #     print("WARNING: idx != res idx", query, outputs)
            
            try:
                candidate_id = node_list[res_idx].key
                if int(res['relevance']) > 0:
                    candidate_ids.add(candidate_id)
                    relevant += 1
            except:
                if strict:
                    assert False, f"Query: {query} returned non-existing node: {res_idx}"
                # else:
                #     print(f"WARNING: Query: {query} returned non-existing node: {res_idx}")
        if verbose:
            print(f"{relevant}/{len(results)} relevant nodes")

        return results, candidate_ids, system_prompt + " " + user, raw_output
    except:
        print("ERROR PARSING JSON: ")
        return [], set(), system_prompt + " " + user, raw_output
        # print(outputs)
        # traceback.print_exc()

def get_goal_candidates_similarity(node_list: List[DialogNode], info_node_embeddings: torch.Tensor, 
                                   query: str, k: int = 10, model: str = "llama3", seed: int = 43, temperature: float = 0.0, 
                                   verbose: bool =False, strict: bool = True,
                                   system_prompt: str = "") -> Tuple[Set[int], List[dict]]:
    # step 1: get most similar nodes
    scores, indices = calculate_similarity(bi_encoder=bi_encoder, query=query, k=k, info_node_embeddings=info_node_embeddings)

    # step 2: use LLM to determine which candidates are useful
    facts = []
    for idx, node_idx in enumerate(indices.tolist()):
        facts.append({
            "id": idx,
            "fact": node_list[node_idx].text
        })
        # print(f"-fact {idx}: {scores[idx]} - {facts[-1]['fact']}")

    user = f"""======= Facts =======
    {json.dumps(facts)}

    ======= Query =======
    "{query}"
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user}
    ]

    outputs = generate_output(model=model, messages=messages, temperature=temperature, seed=seed)
    raw_output = deepcopy(outputs)
    outputs = outputs.replace("\n", "").strip("(").strip(")").strip("'").strip()
    try:
        results = json.loads(outputs)
        all_candidate_ids = set([int(res['id']) for res in results])
        if all_candidate_ids != set(range(len(indices.tolist()))):
            left_diff = len(all_candidate_ids.difference(set(range(len(indices.tolist())))))
            right_diff = len(set(range(len(indices.tolist()))).difference(all_candidate_ids))
            if left_diff > 0:
                if strict:
                    assert False, f"Missing candidate ids from result: {left_diff}"
                else:
                    print("WARNING:", f"Missing candidate ids from result: {left_diff}")
            if right_diff > 0:
                if strict:
                    assert False, f"Candidate ids included in result that are not in node list: {right_diff}"
                else:
                    print("WARNING:", f"Missing candidate ids from result: {right_diff}")

        relevant_candidate_ids = set()
        
        if strict:
            assert len(results) == len(facts)
        for idx, res in enumerate(results):
            res_idx = int(res['id'])
            try:
                candidate_id = node_list[indices[res_idx]].key
                res["node"] = candidate_id
                res["text"] = node_list[indices[res_idx]].text
                if verbose:
                    print(f"-fact idx: {idx}, res_idx: {res_idx}/ {candidate_id}: {scores[idx]} - relevant: {res['relevance']} - {facts[res_idx]['fact']}")
                    print("  -> ", res['justification'])
                if int(res['relevance']) > 0:
                    relevant_candidate_ids.add(candidate_id)
            except:
                if strict:
                    assert False, f"Query: {query} returned non-existing node: {res_idx}"
                else:
                    print(f"WARNING: Query: {query} returned non-existing node: {res_idx}")
        if verbose:
            print(f"{len(relevant_candidate_ids)}/{len(results)} relevant nodes")

        return results, relevant_candidate_ids, system_prompt + " " + user, raw_output
    except:
        # TODO return error message to the user in this case?
        print("ERROR PARSING JSON: ")
        print(outputs)
        traceback.print_exc()
        return [], set(), system_prompt + " " + user, raw_output
    

if PROMPT == 1:
    if USE_JUSTIFICATIONS:
        system_prompt = """You will be provided with a json list of facts and a query.
    You are to act as a first filter to decide which of the given facts answer the query or are relevant to answering the query, at least partially, and which ones are not relevant to answering the query at all.
    Assign each fact a relevance indicator between 0 and 2, and add a justification of why it is relevant (2), partially related (1), or irrelevant (0).
    Facts are also considered relevant if they imply the answer.
    If facts contain placeholders inside curly braces, assume the placeholder will be filled with a reasonable value.
    Don't return anything besides the json list of relevant facts, and only return facts with relevance indicator higher than 0. Don't return code or additional text.

    REMEMBER: even if some facts are only slightly relevant to answering the query, it is better to rate them with a relevance of 1 than to have all facts have relevance 0."""
    else:
        system_prompt = """You will be provided with a json list of facts and a query.
    You are to act as a first filter to decide which of the given facts answer the query or are relevant to answering the query, at least partially, and which ones are not relevant to answering the query at all.
    Assign each fact a relevance indicator between 0 and 2, with relevant (2), partially related (1), or irrelevant (0).
    Facts are also considered relevant if they imply the answer.
    If facts contain placeholders inside curly braces, assume the placeholder will be filled with a reasonable value.
    Don't return anything besides the json list of relevant facts, and only return facts with relevance indicator higher than 0. Don't return code or additional text.

    REMEMBER: even if some facts are only slightly relevant to answering the query, it is better to rate them with a relevance of 1 than to have all facts have relevance 0."""

    if USE_INCONTEXT_EXAMPLES and USE_JUSTIFICATIONS:
        system_prompt +=  """
    
    For example, given the facts:
    [{"key": 0, "fact": "In Singapore, at 9 a.m., it is usually around 35 degrees celsius."},
    {"key": 1, "fact": "In Singapore, between 8 a.m. and 11 a.m., the weather is around 35 degrees celsius."},
    {"key": 2, "fact": "In London, at 9 a.m., it is usually 25 degrees celsius."},
    {"key": 3, "fact": "In Singapore, between 10 a.m. and 11 a.m., it is usually around 30 degrees celsius."},
    {"key": 4, "fact": "In Singapore, there are many tourist attractions."},
    {"key": 5, "fact": In Singapore, it is usually around 35 degrees celsius in the mornings, but cooler in the evenings."},
    {"key": 6, "fact": "In {{ COUNTRY }}, at 9 a.m., it is usually around 35 degrees celsius."},]

    And a query:
    "What is the weather usually in Singapore at 9 a.m.?"

    The reply should only be a json list of the facts, indicating if the facts are related to or directly answering the query, formatted like this:
    [{"key": 0, "relevance": 2, "justification": "The fact is relevant because it answers the user request perfectly"},
    {"key": 1, "relevance": 2, "justification": "The fact is relevant as it answers the user request, because the requested time of 9 a.m. lies between the fact's timespan of 8 a.m. t0 11 a.m."},
    {"key": 2, "relevance": 1, "justification": "While the time is correct, the fact is listing the temperature for London instead of Singapore"},
    {"key": 3, "relevance": 1, "justification": "The fact is talking about the weather in Singapore, which is relevant to the user, alhtough the requested time of 9 a.m. lies outside the fact's timespan of 10 a.m. to 11 a.m."},
    {"key": 5, "relevance": 2, "justification": "The fact is relevant as it partially answers the user query: while it does not state a specific time, it implies the temperatures in Singapore at the requested time"},
    {"key": 6, "relevance": 2, "justification": "The fact is relevant as it could answer the user request perfectly, once the placeholder is filled."},]

    Note that the fact with key 4 was excluded from the output, as it has a relevance of 0: Fact 4 is not related to the query about the weather in Singapore.
    """
    elif USE_INCONTEXT_EXAMPLES and (not USE_JUSTIFICATIONS):
        system_prompt +=  """
    
    For example, given the facts:
    [{"key": 0, "fact": "In Singapore, at 9 a.m., it is usually around 35 degrees celsius."},
    {"key": 1, "fact": "In Singapore, between 8 a.m. and 11 a.m., the weather is around 35 degrees celsius."},
    {"key": 2, "fact": "In London, at 9 a.m., it is usually 25 degrees celsius."},
    {"key": 3, "fact": "In Singapore, between 10 a.m. and 11 a.m., it is usually around 30 degrees celsius."},
    {"key": 4, "fact": "In Singapore, there are many tourist attractions."},
    {"key": 5, "fact": In Singapore, it is usually around 35 degrees celsius in the mornings, but cooler in the evenings."},
    {"key": 6, "fact": "In {{ COUNTRY }}, at 9 a.m., it is usually around 35 degrees celsius."},]

    And a query:
    "What is the weather usually in Singapore at 9 a.m.?"

    The reply should only be a json list of the facts, indicating if the facts are related to or directly answering the query, formatted like this:
    [{"key": 0, "relevance": 2},
    {"key": 1, "relevance": 2},
    {"key": 2, "relevance": 1},
    {"key": 3, "relevance": 1},
    {"key": 5, "relevance": 2},
    {"key": 6, "relevance": 2}]

    Note that the fact with key 4 was excluded from the output, as it has a relevance of 0.
    """

elif PROMPT == 2:
    if USE_JUSTIFICATIONS:
        system_prompt = """You will be provided with a json list of facts and a query.
    You are to act as a first filter to decide which of the given facts answer the query or are relevant to answering the query, at least partially, and which ones are not relevant to answering the query at all.
    Assign each fact a relevance indicator between 0 and 2, and add a justification of why it is relevant (2), partially related (1), or irrelevant (0).
    Facts are also considered relevant if they imply the answer.
    If facts contain placeholders inside curly braces, assume the placeholder will be filled with a reasonable value.
    Each fact should be considered independent of the other facts.
    Don't return anything besides the json list. Don't return code or additional text.

    REMEMBER: even if some facts are only slightly relevant to answering the query, it is better to rate them with a relevance of 1 than to have all facts have relevance 0."""
    else:
        system_prompt = """You will be provided with a json list of facts and a query.
    You are to act as a first filter to decide which of the given facts answer the query or are relevant to answering the query, at least partially, and which ones are not relevant to answering the query at all.
    Assign each fact a relevance indicator between 0 and 2, with relevant (2), partially related (1), or irrelevant (0).
    Facts are also considered relevant if they imply the answer.
    If facts contain placeholders inside curly braces, assume the placeholder will be filled with a reasonable value.
    Each fact should be considered independent of the other facts.
    Don't return anything besides the json list. Don't return code or additional text.

    REMEMBER: even if some facts are only slightly relevant to answering the query, it is better to rate them with a relevance of 1 than to have all facts have relevance 0."""

    if USE_INCONTEXT_EXAMPLES:
        system_prompt +=  """
    
    For example, given the facts:
    [{"key": 0, "fact": "In Singapore, at 9 a.m., it is usually around 35 degrees celsius."},
    {"key": 1, "fact": "In Singapore, between 8 a.m. and 11 a.m., the weather is around 35 degrees celsius."},
    {"key": 2, "fact": "In London, at 9 a.m., it is usually 25 degrees celsius."},
    {"key": 3, "fact": "In Singapore, between 10 a.m. and 11 a.m., it is usually around 30 degrees celsius."},
    {"key": 4, "fact": "In Singapore, there are many tourist attractions."},
    {"key": 5, "fact": In Singapore, it is usually around 35 degrees celsius in the mornings, but cooler in the evenings."},
    {"key": 6, "fact": "In {{ COUNTRY }}, at 9 a.m., it is usually around 35 degrees celsius."},]

    And a query:
    "What is the weather usually in Singapore at 9 a.m.?"

    The reply should only be a json list of the facts, indicating if the facts are related to or directly answering the query, formatted like this:
    [{"key": 0, "relevance": 2, "justification": "The fact is relevant because it answers the user request perfectly"},
    {"key": 1, "relevance": 2, "justification": "The fact is relevant as it answers the user request, because the requested time of 9 a.m. lies between the fact's timespan of 8 a.m. t0 11 a.m."},
    {"key": 2, "relevance": 1, "justification": "While the time is correct, the fact is listing the temperature for London instead of Singapore"},
    {"key": 3, "relevance": 1, "justification": "The fact is talking about the weather in Singapore, which is relevant to the user, alhtough the requested time of 9 a.m. lies outside the fact's timespan of 10 a.m. to 11 a.m."},
    {"key": 4, "relevance": 0, "justification": "The fact is not related to the query about the weather in Singapore"},
    {"key": 5, "relevance": 2, "justification": "The fact is relevant as it partially answers the user query: while it does not state a specific time, it implies the temperatures in Singapore at the requested time"},
    {"key": 6, "relevance": 2, "justification": "The fact is relevant as it could answer the user request perfectly, once the placeholder is filled."},]
    """
else:
    print("ERROR: UNKNOWN PROMPT", PROMPT)


assert MODE in ['train', 'test']
print("MODE", MODE)
print("DATASET", DATA)
print("MODEL", MODEL)
print("PROMPT", PROMPT)
print("PRE-FILTER", USE_PREFILTER)
print("IN-CONTEXT EXAMPLES", USE_INCONTEXT_EXAMPLES)
print("JUSTIFICATIONS", USE_JUSTIFICATIONS)


if DATA == "reimburse":
    data = ReimburseGraphDataset(graph_path=f'en/reimburse/{MODE}_graph.json', answer_path=f'en/reimburse/{MODE}_answers.json', 
                                        use_answer_synonyms=True,
                                        augmentation=DataAugmentationLevel.NONE, augmentation_path=None,
                                        resource_dir="./resources/",
                                        question_limit=0, answer_limit=0, language="en")
elif DATA == "onboarding":
    data = StandardGraphDataset(graph_path=f'en/onboarding/{MODE}_graph.json', answer_path=f'en/onboarding/{MODE}_answers.json', 
                                  use_answer_synonyms=True,
                                  augmentation=DataAugmentationLevel.NONE, augmentation_path=None,
                                  resource_dir="./resources/",
                                  question_limit=0, answer_limit=0, language="en")
elif DATA == "diagnose":
    data = StandardGraphDataset(graph_path=f'en/diagnose/{MODE}_graph.json', answer_path=f'en/diagnose/{MODE}_answers.json', 
                                  use_answer_synonyms=True,
                                  augmentation=DataAugmentationLevel.NONE, augmentation_path=None,
                                  resource_dir="./resources/",
                                  question_limit=0, answer_limit=0, language="en")
else:
    print("ERROR: UNKOWN DATASET")
    exit()


set_seed(seed=SEED)


node_list = get_node_candidate_list_by_type(data=test_data, node_types=NODE_TYPES)
node_text_embeddings = calculate_node_text_embeddings(bi_encoder=bi_encoder, node_list=node_list)

total = 0
found = 0
candidates_per_question = []
times_per_call = []
num_input_tokens_per_question = []
num_output_tokens_per_question = []
gpu_mem_consumption = []
entries_returned = []
errors = 0

for idx, question in tqdm(enumerate(test_data.question_list)):
    if not question.parent.node_type in NODE_TYPES or len(question.text.strip()) == 0:
        continue
    total += 1
    goal_node_key = question.parent.key
    start = time.time()
    try:
        # time retrieval & filtering    
        if USE_PREFILTER:
            all_entries, filtered, input_text, output_text = get_goal_candidates_similarity(node_list=node_list, info_node_embeddings=node_text_embeddings, query=question.text, k=TOP_K, model=MODEL, seed=SEED, temperature=TEMPERATURE, strict=STRICT, system_prompt=system_prompt)
        else:
            all_entries, filtered, input_text, output_text = get_goal_candidates_fullcontext(node_list=node_list, query=question.text, model=MODEL, seed=SEED, temperature=TEMPERATURE, verbose=False, strict=STRICT, system_prompt=system_prompt) 
    except KeyboardInterrupt:
        break
    except:
        print("ERROR:")
        traceback.print_exc()
        all_entries = []
        filtered = []
        errors += 1

    end = time.time()
    time_required = end - start

    # measure GPU utilization
    gpu_mem = get_gpu_utilization() - mem_baseline
    gpu_mem_consumption.append(gpu_mem)

    # count number of input tokens
    q_tokens = len(pipe.tokenizer(input_text)['input_ids'])
    num_input_tokens_per_question.append(q_tokens)

    # count number of output tokens
    a_tokens = len(pipe.tokenizer(output_text)['input_ids'])
    num_output_tokens_per_question.append(a_tokens)

    candidates_per_question.append(len(filtered))
    entries_returned.append(len(all_entries))
    times_per_call.append(time_required)

    if goal_node_key in filtered:
        found += 1
   
    # if idx > 0 and idx % 50 == 0:
    #     print("-------")
    #     print("TOTAL QUESTIONS:", total)
    #     print("FOUND:", found / total)
    #     print("Avg. candidates per node:", mean(candidates_per_question))
    #     print("Avg. time per question:", mean(times_per_call))
    #     print("Avg. #input tokens per question", mean(num_input_tokens_per_question))
    #     print("Avg. #output tokens per question", mean(num_output_tokens_per_question))
print("")
print("===============")
print("TOTAL QUESTIONS:", total)
print("FOUND:", found, "/", total)
print("Accuracy:", found / total)
print("Errors", errors)
print("")
print("Avg. candidates per node:", mean(candidates_per_question))
print("Avg. entries returned per question", mean(entries_returned))
print("Avg. time per question:", mean(times_per_call))
print("")
print("Avg. #input tokens per question", mean(num_input_tokens_per_question))
print("Min #input tokens per question", min(num_input_tokens_per_question))
print("Max #input tokens per question", max(num_input_tokens_per_question))
print("")
print("Avg. #output tokens per question", mean(num_output_tokens_per_question))
print("Min #output tokens per question", min(num_output_tokens_per_question))
print("Max #output tokens per question", max(num_output_tokens_per_question))
print("")
print("MODEL MEM CONSUMPTION", mem_model_consumption)
print("")
print("Avg. GPU mem consumption (w/o models)", mean(gpu_mem_consumption))
print("Min GPU mem consumption (w/o models)", min(gpu_mem_consumption))
print("Max GPU mem consumption (w/o models)", max(gpu_mem_consumption))
print("")
print("Avg. GPU mem consumption (w/ models)", mean([mem + mem_model_consumption for mem in gpu_mem_consumption]))
print("Min GPU mem consumption (w/ models)", min([mem + mem_model_consumption for mem in gpu_mem_consumption]))
print("Max GPU mem consumption (w/ models)", max([mem + mem_model_consumption for mem in gpu_mem_consumption]))

# nvmlShutdown()