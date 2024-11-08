# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ['TRANSFORMERS_CACHE'] = '/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/llm'

# %%
import torch 
import transformers
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, set_seed


# %%
llama = pipeline(
    "text-generation",
    model= "meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.float16},
    device='cuda:0'
)

# %%
# gemma = pipeline(
#     "text-generation",
#     model="google/gemma-2-9b-it",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda:1",
# )
gemma = None

# %%
import openai
from openai import OpenAI

with open("openai_api_key.sh", "r") as f:
    line = f.readline()
    env_var_name, api_key = line.split("=")

client = OpenAI(api_key=api_key)

# %%
from data.dataset import ReimburseGraphDataset, DataAugmentationLevel, NodeType, DialogNode, Answer, NodeType, GraphDataset
from data.parsers.parserValueProvider import ReimbursementRealValueBackend
from environment.goal import UserGoal, GoalPath
import json
from copy import deepcopy
from typing import List, Set, Union, Dict, Tuple
from pprint import pprint
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from collections import defaultdict
from server.nlu import NLU


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
def generate_output(model: str, messages, temperature: float=0.7, seed: int = None, max_new_tokens: int = 10000, batch_size: int = 1) -> str:
    if model in ["gpt-4o", "gpt-4o-mini"]:
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
    elif model in ["llama3", "gemma2"]:
        pipe = llama if model == "llama3" else gemma
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

# %%
# data = ReimburseGraphDataset(graph_path='en/reimburse/train_graph.json', answer_path='en/reimburse/train_answers.json', use_answer_synonyms=False, augmentation=DataAugmentationLevel.NONE)

answerParser = AnswerTemplateParser()
sysParser = SystemTemplateParser()

# %%
from sentence_transformers import SentenceTransformer, CrossEncoder

# %%
bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device="cuda:0", cache_folder="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/")

# %%
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda:0")
cross_encoder = None

# %%
def calculate_info_node_text_embeddings(data: GraphDataset, bi_encoder: SentenceTransformer):
    docs = []
    for node in data.nodes_by_type[NodeType.INFO]:
        docs.append(node.text)
    return bi_encoder.encode(docs, convert_to_tensor=True, batch_size=512)


# %%
def calculate_similarity(bi_encoder: SentenceTransformer, query: str, k: int = 5, info_node_embeddings=None):
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True, batch_size=512)

    similarity_scores = bi_encoder.similarity(query_embedding, info_node_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=k)
    return scores, indices

# %%
def calculate_re_ranking(data: GraphDataset, cross_encoder: CrossEncoder, query: str, k: int = 5):
    docs = []
    for node in data.nodes_by_type[NodeType.INFO]:
        docs.append([query, node.text])
    scores = cross_encoder.predict(sentences=docs, batch_size=512, convert_to_tensor=True)
    scores, indices = scores.topk(k=k)
    return scores, indices

# %%
import traceback


def get_goal_candidates_rerank(data: GraphDataset, query: str, k: int = 10, model: str = "llama3", seed: int = 43, temperature: float = 0.0, verbose: bool =False, strict: bool = True) -> Set[int]:
    # step 1: get most similar nodes
    scores, indices = calculate_re_ranking(data=data, cross_encoder=cross_encoder, query=query, k=k)

    # step 2: use LLM to determine which candidates are useful
    facts = []
    for idx, node_idx in enumerate(indices.tolist()):
        facts.append({
            "id": idx,
            "fact": data.nodes_by_type[NodeType.INFO][node_idx].text
        })
        # print(f"-fact {idx}: {scores[idx]} - {facts[-1]['fact']}")

    user = f"""======= Facts =======
    {json.dumps(facts)}

    ======= Query =======
    "{query}"
    """

    system = """You will be provided with a json list of facts and a query.
    Which of the given facts answer the query or are most closely related, and which ones are unrelated?
    Assign each fact a boolean relevance indicator, and add a justification of why it is relevant or irrelevant.
    Don't return anything besides the json list. Don't return code or additional text.
 
   For example, given the facts:
    [{"key": "12345", "fact": "In Singapore, it is usually around 35 degrees celsius."},
    {"key": "12346", "fact": "In Singapore, the winters are mild and around 20 degrees celsius."},
    {"key": "12347", "fact": "In London, it is usually 25 degrees celsius."},
    {"key": "12348", "fact": "In London, the winters are about 25 degrees celsius."}]

    And a query:
    "How hot is it usually in Singapore?"

    The reply should only be a json list of the facts, indicating if the facts are related to or directly answering the query, formatted like this:
    [{"key": "12345", "relevant": true, "justification": "The fact answers the user request perfectly"},
    {"key": "12346", "relevant": false, "justification": "The fact is talking about winter instead of the user requested general weather"},
    {"key": "12347", "relevant": false, "justification": "The fact is talking about general weather, it is related to London instead of Singapore"},
    {"key": "12348", "relevant": false, "justification": "The fact is talking about London instead of Singapore, and about winter instead of general weather."}]
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    outputs = generate_output(model=model, messages=messages, temperature=temperature, seed=seed)
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
                else:
                    print("WARNING: idx != res idx" )
            
            try:
                candidate_id = data.nodes_by_type[NodeType.INFO][indices[res_idx]].key
                if verbose:
                    print(f"-fact {idx}/{candidate_id}: {scores[idx]} - relevant: {res['relevant']} - {facts[idx]['fact']}")
                    print("  -> ", res['justification'])
                if res['relevant']:
                    candidate_ids.add(candidate_id)
                    relevant += 1
            except:
                if strict:
                    assert False, f"Query: {query} returned non-existing node: {res_idx}"
                else:
                    print(f"WARNING: Query: {query} returned non-existing node: {res_idx}")
        if verbose:
            print(f"{relevant}/{len(results)} relevant nodes")

        return candidate_ids
    except:
        print("ERROR PARSING JSON: ")
        print(outputs)
        traceback.print_exc()

# %%
def get_goal_candidates_similarity(data: GraphDataset, info_node_embeddings: torch.Tensor, query: str, k: int = 10, model: str = "llama3", seed: int = 43, temperature: float = 0.0, verbose: bool =False, strict: bool = True) -> Set[int]:
    # step 1: get most similar nodes
    scores, indices = calculate_similarity(bi_encoder=bi_encoder, query=query, k=k, info_node_embeddings=info_node_embeddings)

    # step 2: use LLM to determine which candidates are useful
    facts = []
    for idx, node_idx in enumerate(indices.tolist()):
        facts.append({
            "id": idx,
            "fact": data.nodes_by_type[NodeType.INFO][node_idx].text
        })
        print(f"-fact {idx}: {scores[idx]} - {facts[-1]['fact']}")

    user = f"""======= Facts =======
    {json.dumps(facts)}

    ======= Query =======
    "{query}"
    """

    system = """You will be provided with a json list of facts and a query.
    Which of the given facts answer the query or are most closely related, and which ones are unrelated?
    Assign each fact a boolean relevance indicator, and add a justification of why it is relevant or irrelevant.
    Don't return anything besides the json list. Don't return code or additional text.
 
   For example, given the facts:
    [{"key": 0, "fact": "In Singapore, at 9 a.m., it is usually around 35 degrees celsius."},
    {"key": 1, "fact": "In Singapore, between 8 a.m. and 11 a.m., the weather is around 35 degrees celsius."},
    {"key": 2, "fact": "In London, at 9 a.m., it is usually 25 degrees celsius."},
    {"key": 3, "fact": "In Singapore, between 10 a.m. and 11 a.m., it is usually around 30 degrees celsius."},
    {"key": 4, "fact": "In Singapore, there are many tourist attractions."}]

    And a query:
    "How hot is it usually in Singapore at 9 a.m.?"

    The reply should only be a json list of the facts, indicating if the facts are related to or directly answering the query, formatted like this:
    [{"key": 0, "relevant": true, "justification": "The fact answers the user request perfectly"},
    {"key": 1, "relevant": true, "justification": "The fact answers the user request, because the requested time of 9 a.m. lies between the fact's timespan of 8 a.m. t0 11 a.m."},
    {"key": 2, "relevant": false, "justification": "While the time is correct, the fact is listing the temperature for London instead of Singapore"},
    {"key": 3, "relevant": false, "justification": "While the fact is talking about the weather in Singapore, the requested time of 9 a.m. lies outside the fact's timespan of 10 a.m. to 11 a.m."},
    {"key": 4, "relevant": false, "justification": "The fact is not related to the query about the weather in Singapore"}]
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    outputs = generate_output(model=model, messages=messages, temperature=temperature, seed=seed)
    outputs = outputs.replace("\n", "").strip("(").strip(")").strip("'").strip()
    print("====")
    print(outputs)
    print("====")
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
                else:
                    print("WARNING: idx != res idx")
            
            try:
                candidate_id = data.nodes_by_type[NodeType.INFO][indices[res_idx]].key
                if verbose:
                    print(f"-fact idx: {idx}, res_idx: {res_idx}/ {candidate_id}: {scores[idx]} - relevant: {res['relevant']} - {facts[res_idx]['fact']}")
                    print("  -> ", res['justification'])
                if res['relevant']:
                    candidate_ids.add(candidate_id)
                    relevant += 1
            except:
                if strict:
                    assert False, f"Query: {query} returned non-existing node: {res_idx}"
                else:
                    print(f"WARNING: Query: {query} returned non-existing node: {res_idx}")
        if verbose:
            print(f"{relevant}/{len(results)} relevant nodes")

        return candidate_ids
    except:
        print("ERROR PARSING JSON: ")
        print(outputs)
        traceback.print_exc()

# %%


# %%
def find_possible_paths(data: GraphDataset,
                        goal_node_ids: List[int],
                        answerParser: AnswerTemplateParser,
                        systemParser: SystemTemplateParser,
                        verbose: bool = False) -> Dict[int, List[GoalPath]]:
    path_map = {}
    for node_id in goal_node_ids:
        node = data.nodes_by_key[node_id]
        goal = UserGoal(data=data, start_node=data.start_node, goal_node=node, initial_user_utterance="", answer_parser=answerParser, system_parser=systemParser, value_backend=None)
        paths = goal.expand_path(goal_node=node, start_node=data.start_node, answerParser=answerParser)
        path_map[node_id] = paths
        if verbose:
            print(f"- Found {len(paths)} possible paths to node: {node_id}")
    
    return path_map

# %%
def trim_paths(paths: List[GoalPath], current_node: DialogNode) -> Dict[int, List[GoalPath]]:
    # trim prefix of path up to (excluding) current node.
    trimmed_paths = {}
    for goal_node in paths:
        trimmed_paths[goal_node] = []
        for path in paths[goal_node]:
            current_node_idx = [node.key for node in path.visited_nodes].index(current_node.key)
            visited_nodes = path.visited_nodes[current_node_idx:]   
            trimmed_paths[goal_node].append(GoalPath(visited_nodes=visited_nodes, visited_ids=path.visited_ids, current_node=None, chosen_answers={}, constraints={}))
    return trimmed_paths 

# %%
def update_and_trim_goals_and_paths(data: GraphDataset, current_node: DialogNode, goal_node_candidates: List[DialogNode], paths: Dict[int, GoalPath], verbose: bool = False):
    # update list of current goal nodes / goal paths based on next node
    next_goal_node_ids = set()
    next_goal_paths = {}
    for goal_node in goal_node_candidates:
        remaining_paths = []
        for path in paths[goal_node.key]:
            if path.visited_nodes[1].key == current_node.key:
                remaining_paths.append(path)
        if len(remaining_paths) > 0:
            next_goal_paths[goal_node.key] = remaining_paths
            next_goal_node_ids.add(goal_node.key)
    next_goal_node_candidates = [data.nodes_by_key[node_id] for node_id in next_goal_node_ids]
    next_goal_paths = trim_paths(paths=next_goal_paths, current_node=current_node)

    if verbose:
        print(f"-- inner REMAINING: Goal nodes: {len(next_goal_node_ids)}, paths: {sum([len(next_goal_paths[nodeid]) for nodeid in next_goal_node_ids])}")
    return next_goal_node_ids, next_goal_node_candidates, next_goal_paths

# %%
from statistics import mean


def get_branches_reaching_all_goal_nodes(current_node: DialogNode, paths: Dict[int, List[GoalPath]], verbose: bool = False) -> Answer:
    """Returns a list of answer branches that each allow reaching all goal nodes."""
    assert current_node.node_type in [NodeType.LOGIC, NodeType.QUESTION]

    shortest_viable_branch = None
    min_avg_branch_length = 0
    for answer in current_node.answers:
        # test for each branch if there exists at least one path in the path list that reaches each goal node via this branch
        if answer.connected_node is None:
            continue # Incomplete tree / no default condition in logic node

        reachable_node_ids = {} # map: goal node id -> shortest path via answer
        for goal_node_id in paths:
            for path in paths[goal_node_id]:
                if path.visited_nodes[1].key == answer.connected_node.key:
                    path_length = len(path.visited_nodes)
                    if not goal_node_id in reachable_node_ids:
                        reachable_node_ids[goal_node_id] = path_length
                    elif path_length < reachable_node_ids[goal_node_id]:
                        reachable_node_ids[goal_node_id] = path_length
        if(len(reachable_node_ids) == len(paths)):
            # each goal node is reachable via the current answer
            # only keep branch with shortest paths (avg) to all nodes
            avg_branch_length = mean(reachable_node_ids.values())
            if shortest_viable_branch is None:
                shortest_viable_branch = answer
                min_avg_branch_length = avg_branch_length
            elif avg_branch_length < min_avg_branch_length:
                shortest_viable_branch = answer
                min_avg_branch_length = avg_branch_length

    if shortest_viable_branch is None:
        if verbose:
            print("COULD NOT FIND VIABLE BRANCH AT NODE", current_node.key, f"({current_node.text[:50]})", "-", current_node.text,)
    return shortest_viable_branch


# %%
def get_most_similar_answer(current_node: DialogNode, query: str) -> Answer:
    assert current_node.node_type == NodeType.QUESTION
 
    # encode & compare query with answer candidates for current node
    answer_candidates = []
    for answer in current_node.answers:
        answer_candidates.append([query, answer.text])
    scores: torch.Tensor = cross_encoder.predict(sentences=answer_candidates, batch_size=len(answer_candidates), convert_to_tensor=True)
    
    # select answer based on highest ranking score
    best_answer_idx = scores.argmax(-1)
    return current_node.answer_by_index(best_answer_idx)

# %%
def best_answer(current_node: DialogNode, user_response: str, model: str = "llama3", seed: int = None, temperature: float = 0.0, verbose: bool = False) -> Answer:
    # shortcut: if there is only one answer, return immediately
    if len(current_node.answers) == 0:
        return current_node.answers[0]

    candidates = []
    for candidate in current_node.answers:
        candidates.append({"index": candidate.index, "text": candidate.text})
    system = f"""Given this list of possible response candidates:
    {json.dumps(candidates)}
    
    Decide which of the reponse candidate texts is most similar to the user input, and only output the most similar response's index as a number.
    Do not giv any other text, any code, or anything else."""

    user = user_response

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    outputs = generate_output(
        model="llama3",
        messages=messages,
        seed=43,
        temperature=0.0)
    if verbose:
        print(" - best answer index:", outputs)
    answer_idx = int(outputs.strip())
    return current_node.answer_by_index(answer_idx)

# %%
def best_answer2(current_node: DialogNode, user_response: str, model: str = "llama3", seed: int = None, temperature: float = 0.0, verbose: bool = False) -> Answer:
    # shortcut: if there is only one answer, return immediately
    if len(current_node.answers) == 0:
        return current_node.answers[0]

    candidates = []
    for candidate in current_node.answers:
        candidates.append({"index": candidate.index, "text": candidate.text})
    example = {"index": 1, "justification": "The user input is a question asking for a comparison between two types of business trips, which is exactly what the response candidate at index 2 is about."}

    system = f"""Given this list of possible response candidates:
    {json.dumps(candidates)}
    
    Decide which of the reponse candidate texts is most similar to the user input, and only output a json object containing the most similar response's index and a short justification of why it is most similar.
    Here is an example of what the output should look like this:
    {json.dumps(example)}
    
    Do not output any other text, any code, or anything else."""

    user = user_response

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    outputs = generate_output(
        model=model, # "llama3"
        messages=messages,
        seed=43,
        temperature=0.0)
    if verbose:
        print(" - best answer index:", outputs)
    outputs = outputs.replace("\n", "").strip("(").strip(")").strip("'").strip()
    result = json.loads(outputs.strip())
    answer_idx = int(result['index'])
    return current_node.answer_by_index(answer_idx)

# %%
from data.parsers.parserValueProvider import ValueBackend


def _eval_logic_node(node: DialogNode, bst: dict, logicParser: LogicTemplateParser, backend: ValueBackend) ->Answer:
    default_branch = None
    for answer in node.answers:
        if answer.text.replace("==", "").replace("}}", "").strip() == "DEFAULT":
            # save default branch for the moment when all branches are evaluated, but none matched
            default_branch = answer
        else:
            condition = node.text + answer.text
            result = logicParser.parse_template(template=condition, backend=backend, bst=bst)
            if result == True:
                return answer
    # no condition matched - follow default branch
    return default_branch




# %%
from environment.cts import CTSEnvironment
from utils.utils import AutoSkipMode
from utils.envutils import GoalDistanceMode

# %%
test_data = ReimburseGraphDataset(graph_path='en/reimburse/test_graph.json', answer_path='en/reimburse/test_answers.json', 
                                  use_answer_synonyms=True,
                                  augmentation=DataAugmentationLevel.NONE, augmentation_path=None,
                                  resource_dir="./resources/",
                                  question_limit=0, answer_limit=0, language="en")

# %%
logicParser = LogicTemplateParser()
answerParser = AnswerTemplateParser()
sysParser = SystemTemplateParser()
value_backend = ReimbursementRealValueBackend(a1_laender=test_data.a1_countries, data=test_data)
# current_node = data.start_node
nlu = NLU()

# %%

def convert_str_to_variable_type(var_name: str, utterance: str, var_type: str):
    # get user reply and save to bst
    if var_name in ["CITY", "COUNTRY"]:
        nlu_results = nlu.extract_places(utterance)
        if var_name in nlu_results and len(nlu_results[var_name]) > 0:
            return nlu_results[var_name][0]
        elif var_name == "CITY":
            return "$REST" # DEFAULT city
    elif var_name == "TRIP_LENGTH":
        nlu_results = nlu.extract_time(utterance)
        return nlu_results['time_spans'][0]
    elif var_name == "PRIVATE_EXTENSION":
        # boolean
        nlu_results = nlu.extract_boolean(utterance)
        return nlu_results[0]
    elif var_type == "NUMBER":
        return float(utterance)
    else:
        return utterance
    # else:
        # print("ERROR PARSING USER INPUT FOR VAR", var_name, "(", var_type, "): ", utterance)


# %%
from enum import Enum

class UserIntent(Enum):
    QUESTION = 1
    STATEMENT = 0

# %%
from config import ActionType
from data.dataset import GraphDataset
from environment.goal import UserInput
from utils.utils import EnvInfo
from data.parsers.parserValueProvider import ValueBackend

class LLMPolicy:
    def __init__(self, data: GraphDataset, env: CTSEnvironment, value_backend: ValueBackend, 
                 model: str, seed: int, temperature: float, top_k: int = 15,
                 verbose: bool = True, strict: bool = True,
                 filter_fn = get_goal_candidates_similarity) -> None:
        self.data = data
        self.model =  model
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.env = env
        self.filter_fn = filter_fn

        self.verbose = verbose
        self.strict = strict

        self.answerParser = AnswerTemplateParser()
        self.logicParser = LogicTemplateParser()
        self.systemParser = SystemTemplateParser()
        self.value_backend = value_backend

        self.turn = 0
        self.free_episode_counter = 0
        self.guided_episode_counter = 0
        self.info_node_embedding = calculate_info_node_text_embeddings(data=data, bi_encoder=bi_encoder)

    # check user intent
    def get_user_intent(self, user_utterance: str) -> UserIntent:
        # return UserIntent.STATEMENT
        messages = [
            {"role": "system", "content": "You classify user input as either question or statement, and only answer with 'question' or 'statement'."},
            {"role": "user", "content": user_utterance},
        ]
        raw = generate_output(model=self.model, messages=messages, seed=self.seed, temperature=self.temperature)
        cleaned = raw.lower()
        assert not ('question' in cleaned and 'statement' in cleaned), f"GOT {cleaned}"
        assert ('question' in cleaned or 'statement' in cleaned) 
        if self.verbose:
            print(f"INTENT: {cleaned}")
        if 'question' in cleaned:
            return UserIntent.QUESTION
        elif 'statement' in cleaned:
            return UserIntent.STATEMENT
        raise Exception(cleaned)

    def _start_dialog_guided(self):
        self.guided_episode_counter += 1


    def _start_dialog_free(self, initial_user_utterance: str, current_node: DialogNode):
        # get the most related goal candidates
        if self.verbose:
            print("INITIAL USER UTTERANCE:", initial_user_utterance)

        k = self.top_k
        self.goal_node_candidate_ids = self.filter_fn(data=self.data, info_node_embeddings=self.info_node_embedding, query=initial_user_utterance, k=k,
                                                           model=self.model, seed=self.seed, temperature=self.temperature,
                                                           verbose=self.verbose, strict=self.strict)
        if self.goal_node_candidate_ids is None:
            print(f"WARNING: NO GOAL RETURN FOR QUERY:", initial_user_utterance)
            self.mode = UserIntent.STATEMENT
            return

        while len(self.goal_node_candidate_ids) == k and k < len(self.data.nodes_by_type[NodeType.INFO]):
            # all TOP-K nodes are related - draw more s.t. we don't miss the user request
            if self.verbose:
                print(f"ALL top {k} entries relevant - doubling K to {k+self.top_k}")
            k += self.top_k
            self.goal_node_candidate_ids = self.filter_fn(data=self.data, info_node_embeddings=self.info_node_embedding, query=initial_user_utterance, k=k,
                                                        model=self.model, seed=self.seed, temperature=self.temperature,
                                                        verbose=self.verbose, strict=self.strict)
            if self.goal_node_candidate_ids is None:
                print(f"WARNING: NO GOAL RETURN FOR QUERY:", initial_user_utterance)
                self.mode = UserIntent.STATEMENT
                return
        self.goal_node_candidates = [self.data.nodes_by_key[key] for key in self.goal_node_candidate_ids]
        
        if len(self.goal_node_candidate_ids) == 0:
            # swtich to guided mode then
            self.mode = UserIntent.STATEMENT
            # if self.verbose:
            print("NO GOAL CANDIDATES FOUND FOR QUERY: " + initial_user_utterance)
            return
        else:
            if self.verbose:
                for idx, goal_node in enumerate(self.goal_node_candidates):
                    print("GOALS:")
                    print(f" {idx}. - {goal_node.key}: {goal_node.text}")
        
        self.free_episode_counter += 1

        # get the paths to all possible goal nodes
        candidate_paths = find_possible_paths(data=self.data, goal_node_ids=self.goal_node_candidate_ids, answerParser=self.answerParser, systemParser=self.systemParser, verbose=self.verbose)
        # trim start node
        self.candidate_paths = trim_paths(
            paths=candidate_paths,
            current_node=current_node)


    def start_dialog(self, initial_user_utterance: str, current_node: DialogNode, intent: UserIntent):
        self.turn = 1
        self.dialog_node_id_history = []
        self.bst = {}
        self.last_sys_act = None
        self.mode = intent
 
        # if self.verbose:
        #     print("GOAL:", self.data.nodes_by_key[self.env.active_env.goal.goal_node_key].text)
        # print(intent, "ENV: ", "free" if self.env.active_env == self.env.free_env else "guided")

        if intent == UserIntent.QUESTION:
            self._start_dialog_free(initial_user_utterance=initial_user_utterance, current_node=current_node)
        else:
            self._start_dialog_guided()

    def _get_next_node_variable(self, current_node: DialogNode) -> int:
        return current_node.answers[0].index + ActionType.SKIP
    
    def _update_bst(self, current_node: DialogNode, user_utterance: str):
        # extract variable
        var_info = self.answerParser.find_variable(current_node.answers[0].text)
        if isinstance(user_utterance, UserInput):
            self.bst[var_info.name] = user_utterance.var_value
        else:
            val = user_utterance.strip()
            # if var_info.type in ["NUMBER", "TIMESPAN", "TIMEPOINT"]:
            #     val = float(val)
            # elif var_info.type == "BOOLEAN":
            #     val = bool(val)
            self.bst[var_info.name] = convert_str_to_variable_type(var_name=var_info.name, utterance=val, var_type=var_info.type)

    def fill_missing_variable(self, var_name: str):
        # track back the current dialog history, and find the last variable node that can fill the variable required by the current node
        for last_node_id in reversed(self.dialog_node_id_history):
            node = self.data.nodes_by_key[last_node_id]
            if node.node_type == NodeType.VARIABLE:
                # extract variable
                var_info = self.answerParser.find_variable(node.answers[0].text)
                if var_info.name == var_name:
                    # ask variable
                    var_value = self.env.request_user_input(node)
                    self.bst[var_info.name] = var_value
                    break
    
    def _get_next_node_logic(self, current_node: DialogNode) -> int:
        # if we know the variable value, we can just evaluate the logic node
        varName = current_node.text.strip("{{").strip()
        if varName in self.bst:
            # evaluate condition
            return _eval_logic_node(node=current_node, bst=self.bst, logicParser=self.logicParser, backend=self.value_backend).index + ActionType.SKIP

        # we don't know the variable value yet:
        # check if logic node is not relevant: if it has a branch that allows reaching all goals
        allreaching_answer = get_branches_reaching_all_goal_nodes(current_node=current_node, paths=self.candidate_paths, verbose=self.verbose)
        if (not allreaching_answer is None):
            return allreaching_answer.index + ActionType.SKIP
        
        # the logic node is decision relevant, since it doesn't have a branch that allows reaching all goals
        # track back the current dialog history, and find the last variable node that can fill the variable required by the current logic node
        self.fill_missing_variable(var_name=varName)
        return self._get_next_node_logic(current_node=current_node)
    
    def _get_next_node_question(self, current_node: DialogNode) -> int:
        allreaching_answer = get_branches_reaching_all_goal_nodes(current_node=current_node, paths=self.candidate_paths, verbose=self.verbose)
        if not allreaching_answer is None:
            # choose any path since all of them lead to all goals
            return allreaching_answer.index + ActionType.SKIP
        else:
            # ask for user input
            return ActionType.ASK

    def _predict_free(self, current_node: DialogNode) -> int:
        if current_node.node_type == NodeType.QUESTION:
            return self._get_next_node_question(current_node=current_node)
        elif current_node.node_type == NodeType.INFO:
            return ActionType.SKIP
        elif current_node.node_type == NodeType.VARIABLE:
            return self._get_next_node_variable(current_node=current_node)
        elif current_node.node_type == NodeType.LOGIC:
            return self._get_next_node_logic(current_node=current_node)
        raise Exception("UNEXPECTED NODE TYPE" + str(current_node))
  
    def _predict_guided(self, current_node: DialogNode) -> int:
        if current_node.node_type == NodeType.QUESTION:
            # ASK - skipping is handled in parent method
            return ActionType.ASK
        elif current_node.node_type == NodeType.INFO:
            # ask node, if we haven't asked it before.
            # otherwise, skip to connected node.
            if self.last_sys_act == ActionType.ASK:
                return ActionType.SKIP
            else:
                return ActionType.ASK
        elif current_node.node_type == NodeType.LOGIC:
            # raise Exception("SHOULD BE HANDLED BY GUIDED ENV" + json.dumps(self.bst))
            return _eval_logic_node(node=current_node, bst=self.bst, logicParser=self.logicParser, backend=self.value_backend).index + ActionType.SKIP
        elif current_node.node_type == NodeType.VARIABLE:
            # ASK - skipping is handled in parent method
            return ActionType.ASK
        
    def _fill_template_variables(self, current_node: DialogNode):
        if current_node.node_type not in [NodeType.INFO, NodeType.QUESTION]:
            return
        
        # mmake sure we have all required variables filled, since this is a text output (ASK) action right now
        var_names = self.systemParser.find_variables(current_node.text)
        for var_name in var_names:
            if not var_name in self.bst:
                self.fill_missing_variable(var_name=var_name)

    
    def predict(self, info: Dict[EnvInfo, any]) -> Tuple[int, UserIntent]:
        current_node_key = info[EnvInfo.DIALOG_NODE_KEY]
        current_node = self.data.nodes_by_key[current_node_key]
        user_utterance = info[EnvInfo.CURRENT_USER_UTTERANCE]
        
        self.turn += 1

        intent = None
        if self.verbose:
            print(f"=== TURN {self.turn} ===")
            print(f"NODE: {current_node_key} ({current_node.node_type}) - {current_node.text[:75]}")
            print(f"USER: {user_utterance}")


        if self.turn == 1:
            # first turn
            intent = self.get_user_intent(user_utterance=info[EnvInfo.INITIAL_USER_UTTERANCE])
            if self.verbose:
                print("INTENT:", intent)
            self.start_dialog(initial_user_utterance=info[EnvInfo.INITIAL_USER_UTTERANCE], current_node=current_node, intent=intent)
        elif self.mode == UserIntent.QUESTION and self.last_sys_act != ActionType.ASK:
            self.goal_node_candidate_ids, self.goal_node_candidates, self.candidate_paths = update_and_trim_goals_and_paths(
                                        data=self.data,
                                        current_node=current_node,
                                        goal_node_candidates=self.goal_node_candidates, paths=self.candidate_paths,
                                        verbose=self.verbose)
        # elif self.last_sys_act == ActionType.ASK:
        #     intent = self.get_user_intent(user_utterance=info[EnvInfo.CURRENT_USER_UTTERANCE])
        self.dialog_node_id_history.append(current_node_key)

        if self.mode == UserIntent.QUESTION and len(self.goal_node_candidate_ids) == 0:
            return -1, None # END DIALOG: no more goal candidates open
        if current_node.connected_node is None and len(current_node.answers) == 0:
            return -1, None # END DIALOG: reached end of tree
        
        if self.last_sys_act == ActionType.ASK and current_node.node_type in [NodeType.QUESTION, NodeType.VARIABLE]:
            if current_node.node_type == NodeType.VARIABLE:
                # we asked for the variable value, now set it and move on to next node
                self._update_bst(current_node=current_node, user_utterance=user_utterance)
                action = ActionType.SKIP
            else:
                # last turn we asked, now we have to skip given the user response
                action = best_answer2(current_node=current_node, user_response=user_utterance, model=self.model, seed=self.seed, temperature=self.temperature, verbose=self.verbose).index + ActionType.SKIP
        elif self.mode == UserIntent.QUESTION and current_node_key in self.goal_node_candidate_ids and self.last_sys_act != ActionType.ASK:
            # ASK the goal once
            action = ActionType.ASK
            if self.verbose:
                print(f"REACHED GOAL: {current_node.text}")
            
            # remove reached goal from candidate list
            self.goal_node_candidate_ids.remove(current_node_key)
            self.goal_node_candidates = [self.data.nodes_by_key[goal_id] for goal_id in self.goal_node_candidate_ids]
            del self.candidate_paths[current_node_key]
        elif self.mode == UserIntent.QUESTION:
            action = self._predict_free(current_node=current_node)
        elif self.mode == UserIntent.STATEMENT:
            action = self._predict_guided(current_node=current_node)
        self.last_sys_act = action
        
        # make sure that template values are filled
        if action == ActionType.ASK:
            self._fill_template_variables(current_node=current_node)

        print("BST", self.bst)
        return action, intent
       
      

# %%
from sklearn.metrics import f1_score

# %%
from dataclasses import dataclass
from typing import Any

from data.parsers.parserValueProvider import RealValueBackend
from environment.base import BaseEnv


@dataclass
class RealUserGoal:
    initial_user_utterance: str
    delexicalised_initial_user_utterance: str
    goal_node_key: str
    constraints: Dict[str, Any]
    visited_ids: Set[int]

    def has_reached_goal_node(self, candidate: DialogNode) -> bool:
        return self.goal_node_key == candidate.key




class RealUserEnvironment(BaseEnv):
    def __init__(self, user_id: int,
            dataset: GraphDataset, nlu: NLU,
            sys_token: str, usr_token: str, sep_token: str,
            max_steps: int, max_reward: float, user_patience: int,
            answer_parser: AnswerTemplateParser, logic_parser: LogicTemplateParser, system_parser: SystemTemplateParser,
            value_backend: RealValueBackend,
            auto_skip: AutoSkipMode, stop_on_invalid_skip: bool, noise: float = 0.0,
            verbose: bool = True,
            auto_skip_logic_nodes: bool = False) -> None:
        assert isinstance(auto_skip, AutoSkipMode)
        super().__init__(dataset=dataset,
            sys_token=sys_token, usr_token=usr_token, sep_token=sep_token, 
            max_steps=max_steps, max_reward=max_reward, user_patience=user_patience,
            answer_parser=answer_parser, logic_parser=logic_parser, value_backend=value_backend,
            auto_skip=auto_skip, stop_on_invalid_skip=stop_on_invalid_skip, noise=noise,
            env_id=user_id, auto_skip_logic_nodes=auto_skip_logic_nodes)
        self.nlu = nlu
        self.verbose = verbose
        self.sys_parser = system_parser

    @property
    def reward_reached_goal(self) -> int:
        return 15

    def reset(self):
        self.pre_reset()

        # Mock a goal node that we can never reach to keep the conversation alive
        # goal_node = DialogNode(key="syntheticGoalNode", text="Synthetic Goal Node", node_type=NodeType.INFO, answers=[], questions=[], connected_node=None)

        # Output first node
        if self.verbose:
            print(self.current_node.text)
        # Ask for initial user input
        initial_user_utterance = deepcopy(input(">>"))
        self.goal = RealUserGoal(initial_user_utterance=initial_user_utterance, delexicalised_initial_user_utterance=initial_user_utterance,
                                 goal_node_key=self.data.start_node.key, constraints=dict(), visited_ids=set())

        return self.post_reset()
    
    def request_user_input(self, node: DialogNode) -> str:
        if node.node_type == NodeType.VARIABLE:
            # get variable name
            var = self.answerParser.find_variable(node.answer_by_index(0).text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 1 # variable value already known
            
            # get user reply and save to bst
            var_value = input(node.text + "\n>>")
            parsed_value = convert_str_to_variable_type(var_name=var.name, utterance=var_value, var_type=var.type)
            self.bst[var.name] = parsed_value 
            self.current_user_utterance = deepcopy(parsed_value)

            self.coverage_variables[var.name][self.bst[var.name]] += 1
        elif node.node_type == NodeType.QUESTION:
            response = input(">>")
            self.current_user_utterance = deepcopy(response)
        return self.current_user_utterance


    def ask(self, replayed_user_utterance: Tuple[str, None]) -> Tuple[bool, float]:
        reward = 0.0
        # output system text
        if self.verbose:
            print("ASKING", sysParser.parse_template(template=self.current_node.text, backend=self.value_backend, bst=self.bst))

        if self.auto_skip_mode != AutoSkipMode.NONE:
            reward -= 1 # because it is 2 actions

        if self.current_node.node_type == NodeType.VARIABLE:
            # get variable name
            var = self.answerParser.find_variable(self.current_node.answer_by_index(0).text)

            # check if variable was already asked
            if var.name in self.bst:
                reward -= 1 # variable value already known
            
            # get user reply and save to bst
            var_value = input(">>")
            parsed_value = convert_str_to_variable_type(var_name=var.name, utterance=var_value, var_type=var.type)
            self.bst[var.name] = parsed_value 
            self.current_user_utterance = deepcopy(parsed_value)

            self.coverage_variables[var.name][self.bst[var.name]] += 1
        elif self.current_node.node_type == NodeType.QUESTION:
            response = input(">>")
            self.current_user_utterance = deepcopy(response)

        return False, reward

    def skip(self, answer_index: int) -> Tuple[bool, float]:
        done = False
        reward = -1.0
        if self.verbose:
            print("SKIPPING WITH ANSWER IDX", answer_index)
        
        next_node = self.get_transition(answer_index)

        if next_node:
            # valid transition
            self.current_node = next_node
            if self.verbose:
                print("-> TO", self.current_node.text[:100])
        else:
            if self.verbose:
                print("next node is None", next_node)
                print("REACHED END OF DIALOG TREE")
            done = True
        return done, reward

    def reached_goal(self) -> Union[bool, float]:
        return False

    def asked_goal(self) -> Union[bool, float]:
        return False



# %%
# How much daily allowance?

# %%
from utils.utils import EnvInfo

# TODO change back
NUM_EPISODES = 1
RENDER = False
VERBOSE = True
STRICT = False
TOP_K = 15
SEED = 43
MODEL = "llama3" # gpt-4o-mini # llama3 # gemma2  # gpt-4o
TEMPERATURE = 0.0
set_seed(SEED)

value_backend = ReimbursementRealValueBackend(a1_laender=test_data.a1_countries, data=test_data)
test_env = RealUserEnvironment(user_id="USER",
                               dataset=test_data, nlu=nlu, sys_token="", usr_token="", sep_token="",
                               max_steps=100, max_reward=100, user_patience=5,
                               answer_parser=answerParser, logic_parser=logicParser, system_parser=sysParser,
                               value_backend=value_backend, auto_skip=AutoSkipMode.NONE,
                               stop_on_invalid_skip=False, noise=0, verbose=True)
policy = LLMPolicy(data=test_data, env=test_env, value_backend=value_backend, model=MODEL,
                   seed=SEED, temperature=TEMPERATURE, top_k=TOP_K, verbose=VERBOSE, strict=STRICT,
                   filter_fn=get_goal_candidates_similarity)

observation = test_env.reset()
episode_starts = True
episode_counts = 0
current_reward = 0
current_length = 0

while episode_counts < NUM_EPISODES:
    try:
        current_node_key = observation[EnvInfo.DIALOG_NODE_KEY]
        current_node = test_data.nodes_by_key[current_node_key]

        action, intent_class = policy.predict(observation)
        if VERBOSE:
            print("ACTION", action)
        if action != -1:
            new_observation, reward, done, = test_env.step(action)
            current_reward += reward
            current_length += 1
        else:
            done = True
            print("DONE")
            # test_env.active_env.dialog_end()

        if done:
            episode_counts += 1

        observation = new_observation
    except KeyboardInterrupt:
        break
   

# %%
1

# %%



