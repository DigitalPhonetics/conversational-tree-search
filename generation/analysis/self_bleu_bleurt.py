# %%
import sys
import os
sys.path.append(os.path.realpath('../../'))

# %%
from tqdm.auto import tqdm
from typing import List
import re
from typing import List, Tuple, Dict
from data.dataset import GraphDataset, ReimburseGraphDataset, DataAugmentationLevel, DialogNode, NodeType
import nltk
from statistics import mean 

# %%
human_data_train = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/train_answers.json', False, augmentation=DataAugmentationLevel.NONE, resource_dir="../../resources/")
human_data_test = ReimburseGraphDataset('en/reimburse/test_graph.json', 'en/reimburse/test_answers.json', False, augmentation=DataAugmentationLevel.NONE, resource_dir="../../resources/")
generated_data_train_v1 = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/generated/train_answers.json', False, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v1.json", resource_dir="../../resources/")
generated_data_train_v2 = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/generated/train_answers.json', False, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v2.json", resource_dir="../../resources/")
generated_data_train_v3 = ReimburseGraphDataset('en/reimburse/train_graph.json', 'en/reimburse/generated/train_answers.json', False, augmentation=DataAugmentationLevel.ARTIFICIAL_ONLY, augmentation_path="en/reimburse/generated/train_questions_v3.json", resource_dir="../../resources/")

# %%
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
chencherry = SmoothingFunction()

def calculate_self_bleu(node: DialogNode, n_grams: int = 3) -> float:
    questions = set([q.text for q in node.questions])
    scores = []
    for hypothesis in questions:
        # take each generated sentence as hypothesis once and test against all other questions as references.
        references = questions.difference(set([hypothesis])) 
        scores.append(sentence_bleu(list(references), hypothesis, smoothing_function=chencherry.method1, weights=[1/n_grams for _ in range(n_grams)]))
    # take average as self-bleu
    return mean(scores)


# %%
datasets = {
    "Human Train": human_data_train,
    # "Human Test": human_data_test,
    "Gen V1": generated_data_train_v1,
    "Gen V2": generated_data_train_v2,
    "Gen V3": generated_data_train_v3
}

# %%
NGRAMS = [1,2,3,4,5]


ngram_scores = {}
for ngram in tqdm(NGRAMS):
    dataset_scores = {}
    for dataset_name in datasets:
        scores = []
        for node in datasets[dataset_name].nodes_by_type[NodeType.INFO]:
            if len(node.questions) <= 1:
                continue
            scores.append(calculate_self_bleu(node, ngram))
        dataset_scores[dataset_name] = mean(scores)
    ngram_scores[ngram] = dataset_scores

# %%
print("BLEU")
print(ngram_scores)


# %% [markdown]
# # Self-BLEURT

# %%
# Install BLEURT: pip install git+https://github.com/google-research/bleurt.git
# GET BLEURT MODEL: wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# %%
from bleurt import score

# %%
checkpoint = "BLEURT-20"
scorer = score.BleurtScorer(checkpoint)

# %%
import itertools

def calculate_self_bleurt(node: DialogNode) -> float:
    questions = set([q.text for q in node.questions])
    scores = []
    # do pair-wise 
    # for reference, hypothesis in itertools.combinations(questions, 2):
    #     score = scorer.score(references=[reference], candidates=[hypothesis])
    #     scores.extend(score)
 
    # do all at once
    # print(mean(scores))
    references, hypotheses = zip(*itertools.combinations(questions, 2))
    scores = scorer.score(references=references, candidates=hypotheses)
    
    return mean(scores)


# %%
print("\n\n\n")
print("BLEURT")
datasets = {
    "Human Train": human_data_train,
    # "Human Test": human_data_test,
    "Gen V1": generated_data_train_v1,
    "Gen V2": generated_data_train_v2,
    "Gen V3": generated_data_train_v3
}


dataset_scores = {}
for dataset_name in tqdm(datasets):
    scores = []
    for node in datasets[dataset_name].nodes_by_type[NodeType.INFO]:
        if len(node.questions) <= 1:
            continue
        scores.append(calculate_self_bleurt(node))
    dataset_scores[dataset_name] = mean(scores)
print(dataset_scores)

# %%



