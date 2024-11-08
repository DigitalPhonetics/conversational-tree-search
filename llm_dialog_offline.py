import os
import traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 
os.environ['TRANSFORMERS_CACHE'] = '/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/llm'


import torch 
import transformers
from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer, set_seed

gemma = pipeline(
    "text-generation",
    model="google/gemma-2-9b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:1",
)

from sklearn.metrics import f1_score
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


def generate_output(model: str, messages, temperature: float=0.7, seed: int = None, max_new_tokens: int = 2048) -> str:
    pipe = gemma
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
        **kwargs,
    )
    raw = outputs[0]["generated_text"][-1]['content']
    return raw

# data = ReimburseGraphDataset(graph_path='en/reimburse/train_graph.json', answer_path='en/reimburse/train_answers.json', use_answer_synonyms=False, augmentation=DataAugmentationLevel.NONE)

answerParser = AnswerTemplateParser()
sysParser = SystemTemplateParser()

from sentence_transformers import SentenceTransformer, CrossEncoder

bi_encoder = SentenceTransformer("multi-qa-mpnet-base-dot-v1", device="cuda:0", cache_folder="/mount/arbeitsdaten/asr-2/vaethdk/resources/weights/")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda:1")


def calculate_info_node_text_embeddings(data: GraphDataset, bi_encoder: SentenceTransformer):
    docs = []
    for node in data.nodes_by_type[NodeType.INFO]:
        docs.append(node.text)
    return bi_encoder.encode(docs, convert_to_tensor=True, batch_size=512)

def calculate_similarity(bi_encoder: SentenceTransformer, query: str, k: int = 5, info_node_embeddings=None):
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True, batch_size=512)

    similarity_scores = bi_encoder.similarity(query_embedding, info_node_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=k)
    return scores, indices

def calculate_re_ranking(data: GraphDataset, cross_encoder: CrossEncoder, query: str, k: int = 5):
    docs = []
    for node in data.nodes_by_type[NodeType.INFO]:
        docs.append([query, node.text])
    scores = cross_encoder.predict(sentences=docs, batch_size=512, convert_to_tensor=True)
    scores, indices = scores.topk(k=k)
    return scores, indices


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
        seed=seed,
        temperature=0.0)
    if verbose:
        print(" - best answer index:", outputs)
    answer_idx = int(outputs.strip())
    return current_node.answer_by_index(answer_idx)


def _get_next_node_question(current_node: DialogNode, paths: Dict[int, List[GoalPath]], verbose: bool = False) -> DialogNode:
    allreaching_answer = get_branches_reaching_all_goal_nodes(current_node=current_node, paths=paths, verbose=verbose)
    if not allreaching_answer is None:
        # choose any path since all of them lead to all goals
        return allreaching_answer.connected_node
    else:
        # ask for user input
        utterance = input(current_node.text)
        # select most similar answer based on user input
        # answer = get_most_similar_answer(current_node=current_node, query=utterance)
        answer = best_answer(current_node=current_node, user_response=utterance, model="llama3", temperature=0, verbose=verbose)
        return answer.connected_node
    
from data.parsers.parserValueProvider import ValueBackend
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

def _eval_logic_node(node: DialogNode, bst: dict, logicParser: LogicTemplateParser, backend: ValueBackend) ->Answer:
    default_branch = None
    for answer in node.answers:
        if answer.text == "== DEFAULT}}":
            # save default branch for the moment when all branches are evaluated, but none matched
            default_branch = answer
        else:
            condition = node.text + answer.text
            result = logicParser.parse_template(template=condition, backend=backend, bst=bst)
            if result == True:
                return answer
    # no condition matched - follow default branch
    return default_branch

def fill_missing_variable(data: GraphDataset, var_name: str, bst: dict, dialog_node_id_history: List[int]) -> dict:
    # track back the current dialog history, and find the last variable node that can fill the variable required by the current node
    for last_node_id in reversed(dialog_node_id_history):
        node = data.nodes_by_key[last_node_id]
        if node.node_type == NodeType.VARIABLE:
            # extract variable
            var_info = answerParser.find_variable(node.answers[0].text)
            if var_info.name == var_name:
                # ask variable
                var_value = input(node.text)
                bst[var_info.name] = var_value
                break
    return bst

def _get_next_node_logic(data: GraphDataset, current_node: DialogNode, paths: Dict[int, List[GoalPath]], bst: dict, logicParser: LogicTemplateParser, answerParser: AnswerTemplateParser, backend: ValueBackend, dialog_node_id_history: List[int]) -> DialogNode:
    # if we know the variable value, we can just evaluate the logic node
    varName = current_node.text.strip("{{").strip()
    if varName in bst:
        # evaluate condition
        return _eval_logic_node(node=current_node, bst=bst, logicParser=logicParser, backend=backend).connected_node

    # we don't know the variable value yet:
    # check if logic node is not relevant: if it has a branch that allows reaching all goals
    allreaching_answer = get_branches_reaching_all_goal_nodes(current_node=current_node, paths=paths)
    if (not allreaching_answer is None):
        return allreaching_answer.connected_node
    
    # the logic node is decision relevant, since it doesn't have a branch that allows reaching all goals
    # track back the current dialog history, and find the last variable node that can fill the variable required by the current logic node
    bst = fill_missing_variable(data=data, var_name=varName, bst=bst, dialog_node_id_history=dialog_node_id_history)
    return _get_next_node_logic(data=data, current_node=current_node, paths=paths, bst=bst, logicParser=logicParser, answerParser=answerParser, backend=backend, dialog_node_id_history=dialog_node_id_history)


def _get_next_node_variable(current_node: DialogNode) -> DialogNode:
    return current_node.answers[0].connected_node

def get_next_node(data: GraphDataset, current_node: DialogNode, paths: Dict[int, List[GoalPath]], bst: dict, logicParser: LogicTemplateParser, answerParser: AnswerTemplateParser, backend: ValueBackend, dialog_node_id_history: List[int], verbose: bool = False) -> DialogNode:
    if current_node.node_type == NodeType.QUESTION:
        return _get_next_node_question(current_node=current_node, paths=paths, verbose=verbose)
    elif current_node.node_type == NodeType.INFO:
        return current_node.connected_node
    elif current_node.node_type == NodeType.VARIABLE:
        return _get_next_node_variable(current_node=current_node)
    elif current_node.node_type == NodeType.LOGIC:
        return _get_next_node_logic(data=data, current_node=current_node, paths=paths, bst=bst, logicParser=logicParser, answerParser=answerParser, backend=backend, dialog_node_id_history=dialog_node_id_history)
    

def get_goal_candidates_similarity_noreasoning(data: GraphDataset, info_node_embeddings: torch.Tensor, query: str, k: int = 10, model: str = "llama3", seed: int = 43, temperature: float = 0.0, verbose: bool =False, strict: bool = True) -> Set[int]:
    # step 1: get most similar nodes
    scores, indices = calculate_similarity(bi_encoder=bi_encoder, query=query, k=k, info_node_embeddings=info_node_embeddings)

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
    Assign each fact a boolean relevance indicator.
    Don't return anything besides the json list. Don't return code or additional text.
 
   For example, given the facts:
    [{"key": "12345", "fact": "In Singapore, it is usually around 35 degrees celsius."},
    {"key": "12346", "fact": "In Singapore, the winters are mild and around 20 degrees celsius."},
    {"key": "12347", "fact": "In London, it is usually 25 degrees celsius."},
    {"key": "12348", "fact": "In London, the winters are about 25 degrees celsius."}]

    And a query:
    "How hot is it usually in Singapore?"

    The reply should only be a json list of the facts, indicating if the facts are related to or directly answering the query, formatted like this:
    [{"key": "12345", "relevant": true},
    {"key": "12346", "relevant": false},
    {"key": "12347", "relevant": false},
    {"key": "12348", "relevant": false}]
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
                    print("WARNING: idx != res idx", query, outputs)
            
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

from environment.cts import CTSEnvironment
from utils.utils import AutoSkipMode
from utils.envutils import GoalDistanceMode

test_data = ReimburseGraphDataset(graph_path='en/reimburse/test_graph.json', answer_path='en/reimburse/test_answers.json', 
                                  use_answer_synonyms=True,
                                  augmentation=DataAugmentationLevel.NONE, augmentation_path=None,
                                  resource_dir="./resources/",
                                  question_limit=0, answer_limit=0, language="en")

logicParser = LogicTemplateParser()
answerParser = AnswerTemplateParser()
sysParser = SystemTemplateParser()
value_backend = ReimbursementRealValueBackend(a1_laender=test_data.a1_countries, data=test_data)
# current_node = data.start_node
nlu = NLU()

from enum import Enum

class UserIntent(Enum):
    QUESTION = 1
    STATEMENT = 0

from config import ActionType
from data.dataset import GraphDataset
from environment.goal import UserInput
from utils.utils import EnvInfo
from data.parsers.parserValueProvider import ValueBackend

class LLMPolicy:
    def __init__(self, data: GraphDataset, env: CTSEnvironment, value_backend: ValueBackend, 
                 model: str, seed: int, temperature: float, top_k: int = 15,
                 verbose: bool = True, strict: bool = True,
                 filter_fn = get_goal_candidates_similarity_noreasoning) -> None:
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
 
        if self.verbose:
            print("GOAL:", self.data.nodes_by_key[self.env.active_env.goal.goal_node_key].text)
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
            if var_info.type in ["NUMBER", "TIMESPAN", "TIMEPOINT"]:
                val = float(val)
            elif var_info.type == "BOOLEAN":
                val = bool(val)
            self.bst[var_info.name] = val

    def fill_missing_variable(self, var_name: str):
        # track back the current dialog history, and find the last variable node that can fill the variable required by the current node
        for last_node_id in reversed(self.dialog_node_id_history):
            node = self.data.nodes_by_key[last_node_id]
            if node.node_type == NodeType.VARIABLE:
                # extract variable
                var_info = self.answerParser.find_variable(node.answers[0].text)
                if var_info.name == var_name:
                    # ask variable
                    var_value = self.env.free_env.goal.get_user_input(node, self.bst, self.data, self.answerParser)
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

        return action, intent
       
    
import numpy as np
from tqdm import tqdm
import random

from utils.utils import EnvInfo

# TODO change back
NUM_EPISODES = 500
RENDER = False
VERBOSE = False
STRICT = False
TOP_K = 15
SEED = 44
MODEL = "gemma2" # gpt-4o-mini # llama3 # gemma2  # gpt-4o
TEMPERATURE = 0.0
set_seed(SEED)

test_env = CTSEnvironment(mode="eval", dataset=test_data, guided_free_ratio=0.5, auto_skip=AutoSkipMode.NONE,
                    normalize_rewards=True, max_steps=50, user_patience=3,
                    stop_when_reaching_goal=True, stop_on_invalid_skip=False,
                    sys_token='SYSTEM:', usr_token='USER:', sep_token='',
                    goal_distance_mode=GoalDistanceMode.FULL_DISTANCE, goal_distance_increment=100,
                    noise=0.0,
                    auto_skip_logic_nodes=False)
value_backend = ReimbursementRealValueBackend(a1_laender=test_data.a1_countries, data=test_data)
policy = LLMPolicy(data=test_data, env=test_env, value_backend=value_backend, model=MODEL,
                   seed=SEED, temperature=TEMPERATURE, top_k=TOP_K, verbose=VERBOSE, strict=STRICT,
                   filter_fn=get_goal_candidates_similarity_noreasoning)

dialog_log = []
error_log = []

episode_rewards_free = []
episode_rewards_guided = []
episode_lengths_free = []
episode_lengths_guided = []
percieved_lengths_free = []
percieved_lengths_guided = []

reached_goals_free = []
reached_goals_guided = []
asked_goals_free = []
asked_goals_guided = []

total_dialogs = 0
free_dialogs = 0
guided_dialogs = 0

intent_episode_log = defaultdict(list)
intent_preds = []
intent_labels = []

episode_counts = 0
current_reward = 0
current_length = 0

observation = test_env.reset()
# observation = test_env.reset()
episode_starts = True

pbar_free = tqdm(total=NUM_EPISODES, desc="Free")
pbar_guided = tqdm(total=NUM_EPISODES, desc="Guided")
while episode_counts < NUM_EPISODES:
    try:
        current_node_key = observation[EnvInfo.DIALOG_NODE_KEY]
        current_node = test_data.nodes_by_key[current_node_key]

        action, intent_class = policy.predict(observation)
        if VERBOSE:
            print("ACTION", action)
        if action != -1:
            new_observation, reward, done, _, info = test_env.step(action)
            current_reward += reward
            current_length += 1
        else:
            done = True
            test_env.active_env.dialog_end()

        if done:
            if VERBOSE:
                print(f"REACHED GOAL: {info[EnvInfo.REACHED_GOAL_ONCE]}")
                print(f"ASKED GOAL: {info[EnvInfo.ASKED_GOAL]}")
                print("####################################")
            # print("DONE")
            # record env mode: free or guided
            total_dialogs += 1
     
            intent_preds.append(policy.mode.value)
            if info[EnvInfo.IS_FAQ]:
                intent_labels.append(UserIntent.QUESTION.value)
                free_dialogs += 1
                episode_rewards_free.append(current_reward)
                episode_lengths_free.append(current_length)
                percieved_lengths_free.append(info[EnvInfo.PERCIEVED_LENGTH])
                dialog_log.extend(test_env.free_env.episode_log)
                asked_goals_free.append(info[EnvInfo.ASKED_GOAL])
                reached_goals_free.append(info[EnvInfo.REACHED_GOAL_ONCE])
                pbar_free.update(1)
            else:
                intent_labels.append(UserIntent.STATEMENT.value)
                guided_dialogs += 1
                episode_rewards_guided.append(current_reward)
                episode_lengths_guided.append(current_length)
                percieved_lengths_guided.append(info[EnvInfo.PERCIEVED_LENGTH])
                dialog_log.extend(test_env.guided_env.episode_log)
                asked_goals_guided.append(info[EnvInfo.ASKED_GOAL])
                reached_goals_guided.append(info[EnvInfo.REACHED_GOAL_ONCE])
                pbar_guided.update(1)
            
            episode_counts += 1
            current_reward = 0
            current_length = 0
            test_env.reset_episode_log()
            policy.turn = 0

            new_observation = test_env.reset()

            if episode_counts % 50 == 0:
                print(f"=== EPISODES: {episode_counts} === ")
                print("Intent F1:", f1_score(y_pred=intent_preds, y_true=intent_labels) )
                print("FREE:", mean(asked_goals_free), "/", mean(reached_goals_free))
                print("GUIDED:", mean(asked_goals_guided), "/", mean(reached_goals_guided))

        observation = new_observation
        if RENDER:
            test_env.render()
    except KeyboardInterrupt:
        break
    except:
        print("======= ERROR ====")
        if info[EnvInfo.IS_FAQ]:
            log = test_env.free_env.episode_log
        else:
            log = test_env.guided_env.episode_log
        error = traceback.format_exc()
        error_log.append({
            "ERROR": error,
            "LOG": log
        })
        test_env.reset_episode_log()
        observation = test_env.reset()
        policy.turn = 0

# LOGS
with open(f"./results/{MODEL}-similarity_k={TOP_K}-errors.json", "w") as f:
    json.dump(error_log, f)
with open(f"./results/{MODEL}-similarity_k={TOP_K}-dialogs.txt", "w") as f:
    f.writelines([log_line + "\n" for log_line in dialog_log])
with open(f"./results/{MODEL}-similarity_k={TOP_K}-stats.json", "w") as f:
    json.dump({
        "summary": {
            "episode_count": episode_counts,
            "dialog_count": {
                "combined": total_dialogs,
                "free": free_dialogs,
                "guided": guided_dialogs
            },
            "intent_f1": f1_score(y_pred=intent_preds, y_true=intent_labels),
            "goals_asked": {
                "combined": mean(asked_goals_free+asked_goals_guided),
                "free": mean(asked_goals_free),
                "guided": mean(asked_goals_guided)    
            },
            "goals_reached": {
                "combined": mean(reached_goals_free+reached_goals_guided),
                "free": mean(reached_goals_free),
                "guided": mean(reached_goals_guided)    
            },
            "episode_lengths": {
                "free": mean(episode_lengths_free),
                "guided": mean(episode_lengths_guided)    
            },
            "perceived_episode_lengths": {
                "free": mean(percieved_lengths_free),
                "guided": mean(percieved_lengths_guided)    
            },
        },
    }, f)

