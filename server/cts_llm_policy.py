from collections import defaultdict
from copy import deepcopy
import json
import logging
import multiprocessing
import pickle
import re
from statistics import mean
import traceback
from typing import Dict, List, Set, Tuple, Union
from enum import Enum


import torch
from tqdm import tqdm
from config import ActionType
from data.dataset import Answer, DialogNode, GraphDataset, NodeType
from data.parsers.answerTemplateParser import AnswerTemplateParser
from data.parsers.logicParser import LogicTemplateParser
from data.parsers.systemTemplateParser import SystemTemplateParser
from environment.goal import GoalPath, UserGoal
from server.nlu import NLU
from data.parsers.parserValueProvider import ValueBackend
from sentence_transformers import SentenceTransformer
import itertools
import re

from openai import OpenAI

class UserIntent(Enum):
    QUESTION = 1
    STATEMENT = 0

url_pattern = re.compile(r'(<a\s+[^>]*href=")([^"]*)(")([^>]*>)')

with open("openai_api_key.sh", "r") as f:
    line = f.readline()
    env_var_name, api_key = line.split("=")
client = OpenAI(api_key=api_key)


# Mono-Lingual
bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1",
                                device="cpu", 
                                cache_folder="./models").eval()




system_1 = """You will be provided with a json list of facts and a query.
You are to act as a first filter to decide which of the given facts answer the query or are relevant to answering the query, at least partially, and which ones are not relevant to answering the query at all?
Assign each fact a relevance indicator between 0 and 2, and add a justification of why it is relevant (2), partially related (1), or irrelevant (0).
Facts are also considered relevant if they imply the answer.
If facts contain placeholders inside curly braces, assume the placeholder will be filled with a reasonable value.
Each fact should be considered independent of the other facts.
Don't return anything besides the json list. Don't return code or additional text.

REMEMBER: even if some facts are only slightly relevant to answering the query, it is better to rate them with a relevance of 1 than to have all facts have relevance 0.

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


def generate_output(model: str, messages, temperature: float=0.7, seed: int = None, max_new_tokens: int = 10000, batch_size: int = 1) -> str:
    if model in ["gpt-4o", "gpt-4o-mini", "gpt-4o-2024-08-06"]:
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
    else:
        raise Exception("UNSUPPORTED MODEL", model)
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

        return relevant_candidate_ids, results
    except:
        # TODO return error message to the user in this case?
        print("ERROR PARSING JSON: ")
        print(outputs)
        traceback.print_exc()
        return set(), {}
    

# Define a function to handle the path expansion logic
# import multiprocessing
# global_data = train_eval_data

# def compute_paths(args):
#     start_node, goal_node = args
#     try:
#         goal = UserGoal(
#             data=global_data,
#             start_node=start_node,
#             goal_node=goal_node,
#             initial_user_utterance="",
#             answer_parser=answerParser,
#             system_parser=sysParser,
#             value_backend=None
#         )
#         paths = goal.expand_path(goal_node=goal_node, start_node=start_node, answerParser=answerParser)
#         if len(paths) > 0:
#             return start_node.key, goal_node.key, paths
#         else:
#             return start_node.key, goal_node.key, []
#     except Exception as e:
#         return start_node.key, goal_node.key, []

# def parallel_path_computation(num_workers=60):
#     path_map = {}

#     # Collect all tasks (start_node, goal_node pairs) to process
#     tasks = []
#     for start_node in global_data.node_list:
#         for goal_node in get_node_candidate_list_by_type(data=global_data, node_types=[NodeType.INFO, NodeType.QUESTION]):
#             tasks.append((start_node, goal_node))

#     # Use multiprocessing Pool to parallelize the task
#     with multiprocessing.Pool(processes=num_workers) as pool:
#         results = list(tqdm(pool.imap(compute_paths, tasks), total=len(tasks)))

#     # Populate the path_map based on results
#     for start_key, goal_key, paths in results:
#         if paths:
#             if not start_key in path_map:
#                 path_map[start_key] = {}
#             if not goal_key in path_map[start_key]:
#                 path_map[start_key][goal_key] = {}
#             path_map[start_key][goal_key] = paths

#     return path_map




def find_possible_paths(data: GraphDataset,
                        start_node: DialogNode,
                        goal_node_ids: List[int],
                        answerParser: AnswerTemplateParser,
                        systemParser: SystemTemplateParser,
                        verbose: bool = False) -> Dict[int, List[GoalPath]]:
    path_map = {}
    for node_id in goal_node_ids:
        node = data.nodes_by_key[node_id]


        try:
            goal = UserGoal(data=data, start_node=start_node, goal_node=node, initial_user_utterance="", answer_parser=answerParser, system_parser=systemParser, value_backend=None)
            paths = goal.expand_path(goal_node=node, start_node=start_node, answerParser=answerParser)
            path_map[node_id] = paths
            if verbose:
                print(f"- Found {len(paths)} possible paths to node: {node_id}")
        except:
            if verbose:
                print(f"- No path found to node: {node_id}")
    
    return path_map

def trim_paths(paths: List[GoalPath], current_node: DialogNode) -> List[GoalPath]:
    # trim prefix of path up to (excluding) current node.
    trimmed_paths = []
    for path in paths:
        path_id_list = [node.key for node in path.visited_nodes]
        if current_node.key in path_id_list:
            current_node_idx = path_id_list.index(current_node.key)
            visited_nodes = path.visited_nodes[current_node_idx:]   
            trimmed_paths.append(GoalPath(visited_nodes=visited_nodes, visited_ids=path.visited_ids, current_node=None, chosen_answers=path.chosen_answers, constraints={}))
    return trimmed_paths 


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

def convert_str_to_variable_type(nlu: NLU, var_name: str, utterance: str, var_type: str):
    # get user reply and save to bst
    if var_name in ["CITY", "COUNTRY"]:
        # nlu_results = nlu.extract_places(utterance)
        # if var_name in nlu_results and len(nlu_results[var_name]) > 0:
        #     return nlu_results[var_name][0]
        # elif var_name == "CITY":
        #     return "$REST" # DEFAULT city
        return utterance
    elif var_name == "TRIP_LENGTH":
        # nlu_results = nlu.extract_time(utterance)
        # return nlu_results['time_spans'][0]
        return float(utterance)
    elif var_name == "PRIVATE_EXTENSION":
        # boolean
        # nlu_results = nlu.extract_boolean(utterance)
        # return nlu_results[0]
        return bool(utterance)
    elif var_type == "NUMBER":
        return float(utterance)
    else:
        return utterance
    # else:
        # print("ERROR PARSING USER INPUT FOR VAR", var_name, "(", var_type, "): ", utterance)
def best_answer4(current_node: DialogNode, user_response: str, model: str = "llama3", seed: int = None, temperature: float = 0.0, verbose: bool = False) -> Answer:
    # shortcut: if there is only one answer, return immediately
    if len(current_node.answers) == 0:
        return current_node.answers[0]

    candidates = []
    for candidate in current_node.answers:
        candidates.append({"index": candidate.index, "text": candidate.text})

    system = f"""Given this list of possible response candidates:
    {json.dumps(candidates)}
    
    Decide which of the reponse candidate texts most closely matches the user intent, and only output the responses index.
    If none of the canidates match, output 'none'.
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
        print(outputs)
    if 'none' in outputs.lower():
        return None
    outputs = re.match(r"(\d+)", outputs).group(1)
    # result = json.loads(outputs.strip())
    answer_idx = int(outputs)
    return current_node.answer_by_index(answer_idx)


class LLMPolicy:
    def __init__(self, user_id: str, socket,  
                 node_list: List[DialogNode], node_embedding: torch.FloatTensor,
                 bi_encoder : SentenceTransformer,
                 path_cache: Dict[int, Dict[int, List[GoalPath]]],
                 data: GraphDataset, 
                 nlu: NLU, sysParser: SystemTemplateParser, answerParser: AnswerTemplateParser, logicParser: LogicTemplateParser,
                 value_backend: ValueBackend, 
                 model: str, seed: int, temperature: float, top_k: int = 15,
                 verbose: bool = True, strict: bool = True,
                 sys_prompt: str = None) -> None:
        self.user_id = user_id
        self.socket = socket
        self.path_cache = path_cache
        self.data = data
        self.model =  model
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k

        self.verbose = verbose
        self.strict = strict

        self.answerParser = answerParser
        self.logicParser = logicParser
        self.systemParser = sysParser
        self.value_backend = value_backend
        self.nlu = nlu

        self.free_episode_counter = 0
        self.guided_episode_counter = 0
        self.current_episode = 0

        self.goal_node_id = None
        self.node_list = node_list
        self.node_embedding = node_embedding
        self.sys_prompt = sys_prompt
        assert sys_prompt is not None and len(sys_prompt) > 0

    def update_and_trim_goals_and_paths(self, current_node: DialogNode):
        # update list of current goal nodes / goal paths based on next node
        next_goal_node_ids = set()
        next_goal_paths = {}
        for goal_node in self.goal_node_candidates:
            remaining_paths = trim_paths(paths=self.candidate_paths[goal_node.key], current_node=current_node)
            # remaining_paths = []
            # for path in self.candidate_paths[goal_node.key]:
            #     if path.visited_nodes[0].key == current_node.key:
            #         remaining_paths.append(path)
            if len(remaining_paths) > 0:
                # node is still reachable via paths left
                next_goal_paths[goal_node.key] = remaining_paths
                next_goal_node_ids.add(goal_node.key)
        next_goal_node_candidates = [self.data.nodes_by_key[node_id] for node_id in next_goal_node_ids]
        # for goal_node_key in next_goal_paths:
        #     next_goal_paths[goal_node_key] = trim_paths(paths=next_goal_paths[goal_node_key], current_node=current_node)

        self.goal_node_candidate_ids = next_goal_node_ids
        self.goal_node_candidates = next_goal_node_candidates
        self.candidate_paths = next_goal_paths

        
    def prefix_exists(self, prefix: List[int], path_candidates: Dict[int, List[GoalPath]]) -> bool:
        # check if there exists a path for each goal node that contains node with "key" at the given position.
        for goal_key in path_candidates:
            found = False
            for path in path_candidates[goal_key]:
                if len(path) >= len(prefix):
                    subpath = [node.key for node in path.visited_nodes[:len(prefix)]]
                    if subpath == prefix:
                        found = True
            if not found:
                return False
        return True

    def get_keys_at_position(self, position: int, path_candidates: Dict[int, List[GoalPath]]) -> Set[int]:
        keys_per_goal = defaultdict(lambda: set())
        for goal_key in path_candidates:
            for path in path_candidates[goal_key]:
                if len(path.visited_nodes) > position:
                    keys_per_goal[goal_key].add(path.visited_nodes[position].key)
        return set.intersection(*list(keys_per_goal.values()))

    def get_longest_shared_prefix(self, path_candidates: Dict[int, List[GoalPath]]) -> List[int]:
        if len(self.goal_node_candidate_ids) == 1:
            # only one prefix -> return 
            last_goal_node_id = list(self.goal_node_candidate_ids)[0]
            shortes_path = self.get_shortest_path(paths=self.candidate_paths[last_goal_node_id])
            return [node.key for node in shortes_path.visited_nodes]
        shared_keys = self.get_keys_at_position(position=0, path_candidates=path_candidates)
        if len(shared_keys) == 0:
            # not even first node idx is shared
            return []
        position = 1
        prefixes = [[key] for key in shared_keys]
        while len(shared_keys) > 0:
            # print(prefixes)
            shared_keys = self.get_keys_at_position(position=position, path_candidates=path_candidates)
            new_prefixes = []
            for prefix in prefixes:
                for key in shared_keys:
                    if self.prefix_exists(prefix + [key], path_candidates):
                        new_prefixes.append(prefix + [key])
            if len(new_prefixes) == 0:
                break
            prefixes = new_prefixes
            position +=1
        return sorted(prefixes, key=lambda prefix: len(prefix), reverse=True)[0]


    def best_answer3(self, current_node: DialogNode, user_response: str) -> Answer:
        # shortcut: if there is only one answer, return immediately
        assert len(current_node.answers) > 0, current_node
        if len(current_node.answers) == 1:
            return current_node.answers[0]

        candidates = []
        for candidate in self.get_node_answer_candidates(current_node=current_node):
            candidates.append({"index": candidate.index, "text": candidate.text})

        system = f"""Given this list of possible response candidates:
        {json.dumps(candidates)}
        
        Decide which of the reponse candidate texts most closely matches the user intent, and only output the responses index.
        Do not output any other text, any code, or anything else."""

        user = user_response

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        outputs = generate_output(
            model=self.model, # "llama3"
            messages=messages,
            seed=self.seed,
            temperature=self.temperature)
        if self.verbose:
            print(outputs)
        try:
            outputs = re.match(r"(\d+)", outputs).group(1)
            # result = json.loads(outputs.strip())
            answer_idx = int(outputs)
            return current_node.answer_by_index(answer_idx)
        except:
            print("Exception: no fitting output for utterance", user_response)
            return current_node.answer_by_index(0)
            # raise Exception(outputs)

    def reached_tree_end(self) -> bool:
        return not self.current_node or (len(self.current_node.answers) == 0 and not self.current_node.connected_node)

    def get_node_markup(self, current_node: DialogNode) -> str:
        # replace links with alert
        markup = url_pattern.sub(r"""\1#\3 onclick="open_link_info()"\4""", current_node.markup)
        try:
            markup = self.systemParser.parse_template(markup, self.value_backend, self.bst)
        except:
            markup = "Sorry, but the dialog system did not ask all neccessary variables. Please consider this dialog ended and proceed with the <b>Finished Dialog</b> button on the right to proceed."
        return markup
    
    def get_shortest_path(self, paths: List[GoalPath]) -> GoalPath:
        shortest_path = None
        shortest_len = None
        for path in paths:
            new_length = len(path.visited_nodes)
            if shortest_path is None:
                shortest_path = path
                shortest_len = new_length
            elif shortest_len > new_length:
                shortest_path = path
                shortest_len = new_length
        return shortest_path

    def get_node_answer_candidates(self, current_node: DialogNode) -> List[Answer]:
        candidates = []
        if self.turn >= 1 and self.mode == UserIntent.QUESTION and current_node.node_type == NodeType.QUESTION:
            # filter list of candidates to options that allow reaching the current goals
            for answer in current_node.answers:
                found_answer = False
                for goal_node_key in self.candidate_paths:
                    for path in self.candidate_paths[goal_node_key]:
                        if current_node.key in path.chosen_answers and answer.key == path.chosen_answers[current_node.key].key:
                            candidates.append(answer)
                            found_answer = True
                            break
                    if found_answer:
                        break
                # TODO should we end the dialog, if none are left?
        elif current_node.node_type == NodeType.VARIABLE:
            var = self.answerParser.find_variable(current_node.answer_by_index(0).text)
            if var.type == "BOOLEAN":
                return [
                    Answer(key=0, text="yes", index=0, parent=current_node, connected_node=None),
                    Answer(key=1, text="no", index=1, parent=current_node, connected_node=None)
                ]
            elif var.type == "LOCATION":
                if var.name == "COUNTRY":
                    return [
                        Answer(key=0, text="USA", index=0, parent=current_node, connected_node=None),
                        Answer(key=1, text="China", index=1, parent=current_node, connected_node=None),
                        Answer(key=2, text="UK", index=2, parent=current_node, connected_node=None),
                        Answer(key=3, text="Germany", index=3, parent=current_node, connected_node=None),
                        Answer(key=4, text="Egypt", index=4, parent=current_node, connected_node=None)
                    ]
            elif var.type == "TIMESPAN":
                return [
                    Answer(key=0, text="2 days", index=0, parent=current_node, connected_node=None),
                    Answer(key=1, text="3 weeks", index=1, parent=current_node, connected_node=None),
                    Answer(key=2, text="1  month", index=2, parent=current_node, connected_node=None)
                ]
        else:
            candidates = current_node.answers

        return candidates # no answer candidates

    # check user intent
    def get_user_intent(self, current_node: DialogNode, user_utterance: str) -> UserIntent:
        if len(user_utterance.strip().split()) == 1:
            return UserIntent.STATEMENT

        answers = []
        for answer in current_node.answers:
            answers.append(f'- "{answer.text}"')
        answers = "\n".join(answers)

        messages = [
            {"role": "system", 
            "content": f"""Answer with "yes" if the text "{user_utterance}" is the same as, a synonym of, or a paraphrase of any of the options given by the user. 
Otherwise return "no":
    """},
            {"role": "user", "content": f""""{answers}"""},
        ]
        raw = generate_output(model=self.model, messages=messages, seed=self.seed, temperature=self.temperature)
        raw = raw.lower().replace("\n", "").replace("```", "").replace("'''", "").replace("json", "").strip("(").strip(")").strip("'").strip()
        if "yes" in raw:
            return UserIntent.STATEMENT
        return UserIntent.QUESTION 
    
    def reachable_goal_node_ids(self, current_node: DialogNode) -> Set[int]:
        if current_node.key in self.path_cache:
            return set(self.path_cache[current_node.key].keys())
        return set()

    def get_goal_nodes(self, user_utterance: str, current_node: DialogNode) -> List[DialogNode]:
        if self.verbose:
            print("Free step: checking for goals")

        # get the most related goal candidates
        if not current_node.key in self.path_cache:
            print(f"WARNING: NO REACHABLE GOALS FROM CURRENT NODE:", user_utterance, current_node.key, current_node.text[:50])
            return None

        # filter out goal candidates that are not reachable 
        reachable_goal_node_ids = self.reachable_goal_node_ids(current_node=current_node)
        reachable_goal_nodes = [self.data.nodes_by_key[goal_id] for goal_id in reachable_goal_node_ids]
        reachable_node_embeddings = self.node_embedding.index_select(0, torch.tensor([self.node_list.index(node) for node in reachable_goal_nodes], dtype=torch.long))
        
        # two-stage filtering: filter reachable goal candidates by similarity, then by LLM reasoning
        goal_node_candidate_ids, _ = get_goal_candidates_similarity(node_list=reachable_goal_nodes, info_node_embeddings=reachable_node_embeddings, query=user_utterance, k=self.top_k,
                                                           model=self.model, seed=self.seed, temperature=self.temperature,
                                                           verbose=self.verbose, strict=self.strict, system_prompt=self.sys_prompt)
        if goal_node_candidate_ids is None:
            print(f"WARNING: NO GOAL RETURN FOR QUERY:", user_utterance)
            return None
   
        goal_node_candidates = [self.data.nodes_by_key[key] for key in goal_node_candidate_ids]
        if len(goal_node_candidate_ids) == 0:
            # swtich to guided mode then
            # if self.verbose:
            print("NO GOAL CANDIDATES FOUND FOR QUERY: " + user_utterance)
            return None
        return goal_node_candidates
  
    def update_goals(self, user_utterance: str, current_node: DialogNode) -> UserIntent:
        # get_user_intent
        # - if statement: continue to next node
        # - if question: get possible goal candidates
        #   - if no goal candidates: guided mode (most similar answer)
        #       - if the goal node is reachable from the current node, continue
        #          - if there are goal candidates left from the previous turns that are still reachable, add them to the candidate list
        #       - otherwise: warn about switching context, restart dialog and move to first decision
        #   - else: move until next question
        # could also run pre-answer decision step with similarity serach: if similarity is very high with one of the answers (e.g. 0.98), classify as statement
        intent = self.get_user_intent(current_node=current_node, user_utterance=user_utterance)
        
        if intent == UserIntent.QUESTION:
            # get possible goal candidates
            goal_candidates = self.get_goal_nodes(user_utterance=user_utterance, current_node=current_node)
            if goal_candidates is None or len(goal_candidates) == 0:
                # no goals found: continue as statement
                if self.verbose:
                    print("- no goals: continuing as statement")
                return UserIntent.STATEMENT
            elif self.turn == 1:
                # add found goal to the goal stack
                # changes = 0
                for goal_node in goal_candidates:
                    if( not goal_node.key in self.goal_node_candidates) and (not goal_node.key in self.visited_goal_ids):
                        # don't visit same goal twice
                        self.goal_node_candidate_ids.add(goal_node.key)
                        self.goal_node_candidates.append(goal_node)
                        self.candidate_paths[goal_node.key] = self.path_cache[current_node.key][goal_node.key]
                # if changes > 0:
                    # we have new goals: calculate longest shared prefix between all goals
                    # self.longest_prefix = self.get_longest_shared_prefix(path_candidates=self.candidate_paths)
        return intent
                
    # TODO: we need something like this, if we want to switch to new questions during the dialog.
    # def get_user_intent_first_node(self, user_utterance: str, model: str, seed: int, temperature: float) -> UserIntent:
    #     messages = [
    #         {"role": "system", 
    #         "content": f"""You are given a list of general statements and a user input.
    #         If the user input is asking a question, or providing more details than any of the intents given, condiser the user input a question.
    #         Otherwise, if the user input matches one of the given general statements, e.g. directly, or as a paraphrase, consider it to be a statement.
    #         Reply with a json object.
            
    #         Example of offered intents:
    #         "Book a trip, Research semester, Travel Risk Management"

    #         Example user inputs and intent classification results:
    #         {
    #             json.dumps([
    #                 {"user": "Plan a trip", "statement": True, "justification": "the user input is a paraphrase of the intent 'book a trip'"},
    #                 {"user": "Book", "statement": True, "justification": "the user input is a paraphrase of the intent 'book a trip'"},
    #                 {"user": "How can I book a flight", "statement": False, "justification": "the user is asking a question that is more specific than the similar intent 'book a trip', because the user specifically needs to book a plane"},
    #                 {"user": "I want to book a flight", "statement": False, "justification": "the user is asking a question that is more specific than the similar intent 'book a trip', because the user specifically needs to book a plane"},
    #                 {"user": "What hotels can I book?", "statement": False, "justification": "the user is asking a question that is more specific than the similar intent 'book a trip', because the user specifically needs to book a hotel"},
    #                 {"user": "There are only double bed rooms left", "statement": False, "justification": "the user is implicitly asking a specific question and is not directly related to, or as general as, any of the given intents"},
    #                 {"user": "Travel safety", "statement": True, "justification": "the user input is a paraphrase of the intent 'travel risk management', is as general as that intent and more a statement than a specific question"},
    #             ])
    #         }
    #         """},
    #         {"role": "user", "content": f"""Given are the following general statements offered by a chatbot:
    #         "{", ".join([a.text for a in self.data.start_node.connected_node.answers])}"
            
    #         And the user input:
    #         "{user_utterance}"
    #         """},
    #     ]
    #     raw = generate_output(model=model, messages=messages, seed=seed, temperature=temperature)
    #     raw = raw.replace("\n", "").replace("```", "").replace("'''", "").replace("json", "").strip("(").strip(")").strip("'").strip()
    #     results = json.loads(raw)
    #     if isinstance(results, list):
    #         for entry in results:
    #             if entry["statement"] == False:
    #                 return UserIntent.QUESTION
    #         return UserIntent.STATEMENT
    #     return UserIntent.STATEMENT if results["statement"] else UserIntent.QUESTION

    def _start_dialog_guided(self):
        self.guided_episode_counter += 1

    def _start_dialog_free(self, initial_user_utterance: str, current_node: DialogNode):
        self.free_episode_counter += 1


    def reset(self, goal_node_id: int):
        # this  should be called before start_dialog
        # only outputs the first system message (greeting) and sets some common state
        logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ RESET')
        if goal_node_id:
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ GOAL ({goal_node_id}): {self.data.nodes_by_key[goal_node_id].text[:75]}')
        elif self.goal_node_id:
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ GOAL ({self.goal_node_id}): {self.data.nodes_by_key[self.goal_node_id].text[:75]}')
        self.turn = 0
        self.done = False
        self.perceived_length = 0
        self.reached_goal_once = False
        self.asked_goal_once = False
        self.dialog_node_id_history = []
        self.goal_node_candidate_ids = set()
        self.goal_node_candidates = []
        self.candidate_paths = {}
        self.longest_prefix = []
        self.visited_goal_ids = set()
        self.current_episode += 1
        self.bst = {}
        self.last_sys_act = ActionType.ASK
        if not goal_node_id is None:
            self.goal_node_id = goal_node_id
        self.current_node = self.data.start_node.connected_node
        self.var_filling_node = None
 
        if self.verbose:
            print("GOAL:", self.data.nodes_by_key[self.goal_node_id].text)
        # print(intent, "ENV: ", "free" if self.env.active_env == self.env.free_env else "guided"
        self.ask(current_node=self.current_node)
        

    def dialog_loop(self, user_utterance: str):
        while not self.done:
            if self.reached_tree_end():
                # output last branch node: this handles the cases where the last node is the goal node 
                # (would otherwise not be asked, because done is already True from here on)
                self.done = True
                self.ask(self.current_node)
                break

            action, first_turn = self.predict(current_node=self.current_node, user_utterance=user_utterance)
            self.last_sys_act = action

            user_utterance = ""
            if action == -2:
                # wait for user input (var / question)
                if not self.var_filling_node is None:
                    self.ask(current_node=self.var_filling_node)
                else:
                    self.ask(current_node=self.current_node)
                return
            elif action == -1:
                # policy thinks the dialog is over (e.g., no more goal candidates left) - end dialog
                self.done = True
            elif action == ActionType.ASK:
                # output current node text
                self.ask(self.current_node)
            else:
                # skip
                self.skip(action=action, first_turn=first_turn)

        logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ BST: {json.dumps(self.bst)}')
        if self.done:
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ REACHED GOAL ONCE: {self.reached_goal_once}')
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ ASKED GOAL ONCE: {self.asked_goal_once}')
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ TOTAL LENGTH: {self.turn}')
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ PERCEIVED LENGTH: {self.perceived_length}')
            self.socket.write_message({"EVENT": "DIALOG_ENDED", "VALUE": True})

    def ask(self, current_node: DialogNode):
        # this action writes output to the user
        logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ SYSTEM (ASK, {current_node.key}): {current_node.text[:75]}')
        
        if self.current_node.key == self.data.start_node.connected_node.key and self.dialog_node_id_history.count(current_node.key) >= 1:
            # this is the second time we see the start node - reformat slightly to encourage user to choose one of the available options
            options = "".join( [f"<li>{cand.text}</li>" for cand in self.get_node_answer_candidates(current_node=current_node)])
            self.socket.write_message({"EVENT": "MSG",
                                        "VALUE": f"I found answers to your question in the following categories, please select the category that matches your situation:<br><br><ul>{options}</ul>", 
                                        "CANDIDATES":[], # self.get_node_answer_candidates(current_node=current_node), 
                                        "NODE_TYPE": current_node.node_type.value  })
        else:
            self.socket.write_message({"EVENT": "MSG",
                                        "VALUE": self.get_node_markup(current_node=current_node), 
                                        "CANDIDATES": [cand.text for cand in self.get_node_answer_candidates(current_node=current_node)], 
                                        "NODE_TYPE": current_node.node_type.value  })
        self.perceived_length += 1
        if current_node.key in self.goal_node_candidate_ids:
            # remove current node from goal stack
            self.goal_node_candidate_ids.remove(current_node.key)
            self.goal_node_candidates = [cand for cand in self.goal_node_candidates if cand.key != current_node.key]
            del self.candidate_paths[current_node.key]
        if current_node.key == self.goal_node_id: # don't use self.current_node, because this could be a place where we fill missing variables
            self.asked_goal_once = True
            self.done = True
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ ASKED GOAL ONCE')

    def skip(self, action: int, first_turn: bool):
        # this action changes the current node

        # subtract ASk-action to get the real answer skip index
        skip_idx = action - 1
        assert action >= 0

        if not first_turn:
            if self.current_node.node_type in [NodeType.INFO, NodeType.VARIABLE_UPDATE]:
                assert skip_idx == 0
                # move to connected neighbor
                self.current_node = self.current_node.connected_node
                logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ SYSTEM (SKIP, {self.current_node.key}): {self.current_node.text[:75]}')
            else:
                if self.current_node.node_type == NodeType.VARIABLE:
                    assert skip_idx == 0
                else:
                    assert skip_idx < len(self.current_node.answers)
                self.current_node = self.current_node.answer_by_index(skip_idx).connected_node
                logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ SYSTEM (SKIP, {self.current_node.key}): {self.current_node.text[:75]}')
            if self.current_node.key in self.goal_node_candidate_ids:
                self.reached_goal_once = True
                self.visited_goal_ids.add(self.current_node.key)
                logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ REACHED GOAL ONCE')
                return

        self.update_and_trim_goals_and_paths(current_node=self.current_node)
        if self.mode == UserIntent.QUESTION:
            self.current_node = self.jump_to_end(current_node=self.current_node)
        self.update_and_trim_goals_and_paths(current_node=self.current_node)
        
        if self.current_node.key == self.goal_node_id:
            self.reached_goal_once = True
            self.visited_goal_ids.add(self.current_node.key)
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ REACHED GOAL ONCE')

    # TODO adapt
    def user_reply(self, msg):
        # called when user input is available
        if self.done:
            return

        logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ USER: {msg}')

        if (not self.var_filling_node is None) or self.last_sys_act == -2:
            # fill variable node 
            node = self.current_node if self.var_filling_node is None else self.var_filling_node
            if node.node_type == NodeType.VARIABLE:
                error = self.check_and_set_variable(node=node, utterance=msg)
                if not error is None:
                    # send error to user 
                    logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ VAR ERROR: {error}')
                    self.socket.write_message({"EVENT": "MSG", "VALUE": error, "NODE_TYPE": node.node_type.value})
                    return
                else:
                    # continue dialog (w/o user utterance, was already used to fill variable)
                    self.var_filling_node = None
                    msg = ""

        # proceed with dialog system loop
        self.dialog_loop(user_utterance=msg)

    def set_inital_variables(self, initial_utterance: str):
        places_results = self.nlu.extract_places(initial_utterance)
        if "CITY" in places_results and len(places_results["CITY"]) == 1:
            self.bst["CITY"] = "$REST"
        if "COUNTRY" in places_results and len(places_results["COUNTRY"]) == 1:
            self.bst["COUNTRY"] = places_results["COUNTRY"][0]
        
        time_results = self.nlu.extract_time(initial_utterance)
        if "time_spans" in time_results and len(time_results['time_spans']) == 1:
            self.bst["TRIP_LENGTH"] = time_results['time_spans'][0]


    def check_and_set_variable(self, node: DialogNode, utterance: str) -> Union[str, None]:
        # Retuns error string if problem
        # else return None, and updates the BST

        # in UI, check if error string not None 
        #   -> step, if None
        #   -> don't step, write error msg if not None
        
        # get variable name
        var = self.answerParser.find_variable(node.answer_by_index(0).text)
        if var.name in ["CITY", "COUNTRY"]:
            nlu_results = self.nlu.extract_places(utterance)
            if var.name in nlu_results:
                if len(nlu_results[var.name]) > 1:
                    return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results[var.name])}"
                if len(nlu_results[var.name]) == 0:
                    if var.name == "COUNTRY":
                        return f"Sorry, but the {var.name.lower()} you entered is unknown to the system. Please check spelling or try another value."
                    elif var.name == "CITY":
                        self.bst[var.name] = "$REST"
                        return None # unkown cities will default to $REST
            self.bst[var.name] = nlu_results[var.name][0]
        elif var.name == "TRIP_LENGTH":
            nlu_results = self.nlu.extract_time(utterance)
            if len(nlu_results['time_spans']) > 1:
                return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results['time_spans'])}"
            elif len(nlu_results['time_spans']) == 0:
                return f"Sorry, but the time span you entered was not recognized by the system. Please rephrase."
            self.bst[var.name] = nlu_results["time_spans"][0]
        elif var.name == "PRIVATE_EXTENSION":
            # boolean
            nlu_results = self.nlu.extract_boolean(utterance)
            if len(nlu_results) > 1:
                return f"Please provide only a single value. Detected multiple values: {', '.join(nlu_results)}" 
            elif len(nlu_results) == 0:
                return f"Sorry, but the value you provided could not be interpreted as confirmation nor the opposite. Please try to rephrase."
            self.bst[var.name] = nlu_results[0]
        else:
            return "ERROR: Found unknown variable. Please report this problem."
        
        return None

    def _get_next_node_variable(self, current_node: DialogNode) -> int:
        return current_node.answers[0].index + ActionType.SKIP

    def fill_missing_variable(self, var_name: str) -> bool:
        # track back the current dialog history, and find the last variable node that can fill the variable required by the current node
        for last_node_id in reversed(self.dialog_node_id_history):
            node = self.data.nodes_by_key[last_node_id]
            if node.node_type == NodeType.VARIABLE:
                # extract variable
                var_info = self.answerParser.find_variable(node.answers[0].text)
                if var_info.name == var_name:
                    self.var_filling_node = node
                    # WAIT FOR USER INPUT
                    return True
        return False

    def _fill_template_variables(self, current_node: DialogNode) -> bool:
        if current_node.node_type not in [NodeType.INFO, NodeType.QUESTION]:
            return False
        
        # make sure we have all required variables filled, since this is a text output (ASK) action right now
        var_names = self.systemParser.find_variables(current_node.text)
        for var_name in var_names:
            if not var_name in self.bst:
                if self.fill_missing_variable(var_name=var_name):
                    return True
        return False
    
    def _get_next_node_logic(self, current_node: DialogNode) -> int:
        # if we know the variable value, we can just evaluate the logic node
        varName = current_node.text.strip("{{").strip()
        if varName in self.bst:
            # evaluate condition
            return _eval_logic_node(node=current_node, bst=self.bst, logicParser=self.logicParser, backend=self.value_backend).index + ActionType.SKIP

        # the logic node is decision relevant, since it doesn't have a branch that allows reaching all goals
        # track back the current dialog history, and find the last variable node that can fill the variable required by the current logic node
        assert self.fill_missing_variable(var_name=varName)
        return -2
  
    def _get_next_node_variable_update(self, current_node: DialogNode) -> int:
        # get variable and type
        pattern = r'\s*(\w+)\s*\((\w+)\)\s*:=\s*(\w+)'
        match = re.match(pattern, current_node.text)
        var_name, var_type, var_value = match.groups()

        # update bst
        if var_type == "BOOLEAN":
            assert var_value.lower() in ["true", "false"]
            self.bst[var_name] = True if var_value.lower() == "true" else False
        # TODO support more variable types

        # move on to next node
        return ActionType.SKIP # only 1 connected node, no answers

    def _predict_guided(self, current_node: DialogNode) -> int:
        if current_node.node_type == NodeType.QUESTION:
            # ASK - skipping is handled in parent method
            return ActionType.ASK
        elif current_node.node_type == NodeType.INFO:
            # ask node, if we haven't asked it before.
            # otherwise, skip to connected node.
            if self.last_sys_act in [ActionType.ASK, -2]:
                return ActionType.SKIP
            else:
                return ActionType.ASK
        elif current_node.node_type == NodeType.LOGIC:
            # raise Exception("SHOULD BE HANDLED BY GUIDED ENV" + json.dumps(self.bst))
            return _eval_logic_node(node=current_node, bst=self.bst, logicParser=self.logicParser, backend=self.value_backend).index + ActionType.SKIP
        elif current_node.node_type == NodeType.VARIABLE:
            # ASK - skipping is handled in parent method
            return ActionType.ASK
        elif current_node.node_type == NodeType.VARIABLE_UPDATE:
            return self._get_next_node_variable_update(current_node=current_node)
        raise Exception("UNEXPECTED NODE TYPE" + str(current_node))

    def jump_to_end(self, current_node: DialogNode) -> DialogNode:
        # calculate longest common path prefix & jump to end
        # filter out reachable nodes
        # then, re-calculate longest path prefix
        if self.mode == UserIntent.STATEMENT or len(self.goal_node_candidate_ids) == 0:
            # guided mode, no available goal nodes
            return current_node

        # self.candidate_paths = find_possible_paths(data=self.data, start_node=current_node, goal_node_ids=self.goal_node_candidate_ids,
        #                                             answerParser=self.answerParser, systemParser=self.systemParser, verbose=self.verbose)
        self.longest_prefix = self.get_longest_shared_prefix(path_candidates=self.candidate_paths)
        if len(self.longest_prefix) == 0:
            return current_node

        self.last_sys_act = ActionType.SKIP
        self.dialog_node_id_history.extend(self.longest_prefix)
        next_node = self.data.nodes_by_key[self.longest_prefix[-1]]
        # self.candidate_paths = {}
        # for goal_key in self.goal_node_candidate_ids:
        #     self.candidate_paths[goal_key] = self.path_cache[current_node.key][goal_key] 
        return next_node

    def predict(self, current_node: DialogNode, user_utterance: str) -> Tuple[int, UserIntent]:
        self.turn += 1

        if self.turn == 1:
            # first turn
            self.set_inital_variables(initial_utterance=user_utterance)

        # Intent tracker & update goal stack
        intent = None
        if (not user_utterance is None) and len(user_utterance.strip()) > 0 and self.turn == 1:
            intent = self.update_goals(user_utterance=user_utterance, current_node=current_node) # update goals
            logging.getLogger("chat").info(f'{self.user_id}-{self.current_episode}$ INTENT: {intent.value}') 
            if len(self.goal_node_candidate_ids) == 0 and intent == UserIntent.STATEMENT:
                self.mode = UserIntent.STATEMENT
            else:
                self.mode = UserIntent.QUESTION

        if self.turn > 1 and self.mode == UserIntent.QUESTION and len(self.goal_node_candidate_ids) == 0:
            # end dialog, no goals left
            return -1, None 
        elif self.turn == 1 and self.mode == UserIntent.QUESTION:
            return ActionType.SKIP, True
        
        if current_node.key in self.goal_node_candidate_ids and not self.last_sys_act == ActionType.ASK:
            # current node is goal node, was not shown yet - output to the user
            if self._fill_template_variables(current_node=current_node):
                # fill missing variables
                return -2, None
            return ActionType.ASK, None


        # now, we are at a decision node
        if current_node.node_type == NodeType.LOGIC:
            action_idx = self._get_next_node_logic(current_node=current_node)
            # ask missing variable, or skip to followup-node
            return action_idx, None
        elif current_node.node_type == NodeType.QUESTION:
            if self._fill_template_variables(current_node=current_node):
                # fill missing variables
                return -2, None
            elif not self.last_sys_act in [-2, ActionType.ASK]:
                # output question
                return -2, None
            else:
                # move to next node
                skip_idx = self.best_answer3(current_node=current_node, user_response=user_utterance).index + ActionType.SKIP
                return skip_idx, None
        elif current_node.node_type == NodeType.INFO:
            if self._fill_template_variables(current_node=current_node):
                # fill missing variables
                return -2, None
            elif not self.last_sys_act in [-2, ActionType.ASK]:
                # output question
                return ActionType.ASK, None
            else:
                return ActionType.SKIP, None
        elif current_node.node_type == NodeType.VARIABLE:
            var = self.answerParser.find_variable(current_node.answer_by_index(0).text)
            if var.name in self.bst:
                return ActionType.SKIP, None
            else:
                return -2, None
        
        print("PROBLEM")
        # # make sure that template values are filled
        # if action == ActionType.ASK:
        #     if self._fill_template_variables(current_node=current_node):
        #         return -2, None
        #     if self.current_node.node_type in [NodeType.QUESTION, NodeType.VARIABLE]:
        #         return -2, None
        
        # if self.verbose:
        #     print("BST", self.bst)

        # return action, intent
       
      