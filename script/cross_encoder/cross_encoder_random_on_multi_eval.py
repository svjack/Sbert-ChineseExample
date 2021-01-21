#!/usr/bin/env python
# coding: utf-8
#es_host = 'localhost:9200'
import csv
import gzip
import json
import logging
import os
import tarfile
import time
from datetime import datetime
#from es_pandas import es_pandas
from functools import reduce
from glob import glob
from typing import Callable, Dict, Iterable, List, Type

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import transformers
#from elasticsearch import Elasticsearch, helpers
from sentence_transformers import (InputExample, LoggingHandler,
                                   SentenceTransformer, util)
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers.evaluation import (SentenceEvaluator,
                                              SequentialEvaluator)
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)


logger = logging.getLogger(__name__)
pd.set_option("display.max_rows", 200)


class DictionaryEvaluator(SequentialEvaluator):
    def __init__(self, evaluators: Iterable[SentenceEvaluator], main_score_function = lambda x: x):
        super(DictionaryEvaluator, self).__init__(evaluators, main_score_function)
        #self.eval_name_ext_dict = dict(map(lambda t2: (t2[0].lower()[:t2[0].lower().find("Evaluator".lower())], t2[1]) ,map(lambda eval_ext: (eval_ext.name, eval_ext) ,self.evaluators)))
        self.eval_name_ext_dict = dict(map(lambda t2: (t2[0], t2[1]) ,map(lambda eval_ext: (eval_ext.name, eval_ext) ,self.evaluators)))
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        scores = {}
        #for evaluator in self.evaluators:
        for eval_ext_name, evaluator in self.eval_name_ext_dict.items():
            #scores.append(evaluator(model, output_path, epoch, steps))
            #eval_output_path = output_path + eval_ext_name
            eval_output_path = output_path + "_" + eval_ext_name
            #scores[eval_ext_name] = evaluator(model, output_path, epoch, steps)
            if eval_output_path is not None and not(os.path.exists(eval_output_path)):
                os.makedirs(eval_output_path, exist_ok=True)
            scores[eval_ext_name] = evaluator(model, eval_output_path, epoch, steps)
        return self.main_score_function(scores)

class CrossEncoder_Dict_Eval(CrossEncoder):
    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            if isinstance(evaluator, DictionaryEvaluator):
                if type(self.best_score) != type({}):
                    self.best_score = dict(map(lambda eval_ext_name: (eval_ext_name, self.best_score), evaluator.eval_name_ext_dict.keys()))
                score_dict = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
                if callback is not None:
                    callback(score, epoch, steps)
                for eval_ext_name, eval_score in score_dict.items():
                    if eval_score > self.best_score[eval_ext_name]:
                        self.best_score[eval_ext_name] = eval_score
                        if save_best_model:
                            #eval_output_path = output_path + eval_ext_name
                            eval_output_path = output_path + "_" + eval_ext_name
                            self.save(eval_output_path)
            else:
                score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
                if callback is not None:
                    callback(score, epoch, steps)
                if score > self.best_score:
                    self.best_score = score
                    if save_best_model:
                        self.save(output_path)


class CERerankingEvaluatorSUM:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, mrr_at_k: int = 10, name: str = '', num_dev_queries = 600):
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        
        self.num_dev_queries = num_dev_queries

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())
        #output/training_ms-marco_cross-encoder-xlm-roberta-base-2021-01-12_21-10-39mrr-train-eva/CERerankingEvaluator_mrr-train-eval_results.csv
        self.csv_file = "CERerankingEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MRR@{}".format(mrr_at_k)]
        
        self.score_json_file = self.csv_file.replace(".csv", ".json")

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CERerankingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_mrr_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        scores_list = []
        
        samples = list(self.samples)
        samples_indexes = np.random.permutation(np.arange(len(samples)))
        samples = list(map(lambda idx: samples[idx], samples_indexes[:self.num_dev_queries]))
        #for instance in self.samples:
        for instance in samples:
            query = instance['query']
            positive = list(instance['positive'])
            negative = list(instance['negative'])
            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            scores_list.extend(list(pred_scores))
            pred_scores_argsort = np.argsort(-pred_scores)  #Sort in decreasing order

            mrr_score = 0
            for rank, index in enumerate(pred_scores_argsort[0:self.mrr_at_k]):
                if is_relevant[index]:
                    mrr_score += 1 / (rank+1)

            all_mrr_scores.append(mrr_score)

        mean_mrr = np.mean(all_mrr_scores)
        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives), np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)))
        logger.info("MRR@{}: {:.2f}".format(self.mrr_at_k, mean_mrr*100))

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                writer.writerow([epoch, steps, mean_mrr])
            json_path = os.path.join(output_path, self.score_json_file)
            output_file_exists = os.path.isfile(json_path)
            with open(json_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(["epoch", "steps", "score@"])
                writer.writerow([epoch, steps, json.dumps({"scores_list": list(map(float ,scores_list))})])
        return mean_mrr

class ScoreCalculator(object):
    def __init__(self,
                queries_ids,
                relevant_docs,
                mrr_at_k: List[int] = [10],
                ndcg_at_k: List[int] = [10],
                accuracy_at_k: List[int] = [1, 3, 5, 10],
                precision_recall_at_k: List[int] = [1, 3, 5, 10],
                map_at_k: List[int] = [100],
    ):
        "queries_ids list of query, relevant_docs key query value set or list of relevant_docs"
        self.queries_ids = queries_ids
        self.relevant_docs = relevant_docs
        
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
    def compute_metrics(self, queries_result_list: List[object]):
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        #### elements with hits one hit is a dict : {"corpus_id": corpus_text, "score": score}
        #### corpus_id replace by corpus text
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores
            top_hits = sorted(queries_result_list[query_itr], key=lambda x: x['score'], reverse=True)
            query_relevant_docs = self.relevant_docs[query_id]

            # Accuracy@k - We count the result correct, if at least one relevant doc is accross the top-k documents
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [1 if top_hit['corpus_id'] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(true_relevances, k_val)
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit['corpus_id'] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)

                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            #num_hits_at_k[k] /= len(self.queries)
            num_hits_at_k[k] /= len(queries_result_list)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            #MRR[k] /= len(self.queries)
            MRR[k] /= len(queries_result_list)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])
        return {'accuracy@k': num_hits_at_k, 'precision@k': precisions_at_k, 'recall@k': recall_at_k, 'ndcg@k': ndcg, 'mrr@k': MRR, 'map@k': AveP_at_k}
    @staticmethod
    def compute_dcg_at_k(relevances, k):
        dcg = 0
        for i in range(min(len(relevances), k)):
            dcg += relevances[i] / np.log2(i + 2)  #+2 as we start our idx at 0
        return dcg

def map_dev_samples_to_score_calculator_format(dev_samples):
    if isinstance(dev_samples, dict):
        dev_samples = list(dev_samples.values())
    queries_ids = list(map(lambda x: x["query"] ,dev_samples))
    relevant_docs = dict(map(lambda idx: (dev_samples[idx]["query"], dev_samples[idx]["positive"]), range(len(dev_samples))))
    return ScoreCalculator(queries_ids, relevant_docs)

class ScoreEvaluator:
    """
    This class evaluates a CrossEncoder model for the task of re-ranking.

    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 is compute to measure the quality of the ranking.

    :param samples: Must be a list and each element is of the form: {'query': '', 'positive': [], 'negative': []}. Query is the search query,
     positive is a list of positive (relevant) documents, negative is a list of negative (irrelevant) documents.
    """
    def __init__(self, samples, name: str = '', num_dev_queries = 600):
        self.samples = samples
        self.name = name
        self.num_dev_queries = num_dev_queries

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())
            
        #self.score_calculator_ext = map_dev_samples_to_score_calculator_format(self.samples)
        
        self.csv_file = "ScoreEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "MAP@"]
        
        self.score_json_file = self.csv_file.replace(".csv", ".json")

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("ScoreEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        all_scores = []
        num_queries = 0
        num_positives = []
        num_negatives = []
        
        scores_list = []
        
        samples = list(self.samples)
        samples_indexes = np.random.permutation(np.arange(len(samples)))
        samples = list(map(lambda idx: samples[idx], samples_indexes[:self.num_dev_queries]))
        #for instance in self.samples:
        for instance in samples:
            query = instance['query']
            positive = list(instance['positive'])
            negative = list(instance['negative'])
            docs = positive + negative
            is_relevant = [True]*len(positive) + [False]*len(negative)

            if len(positive) == 0 or len(negative) == 0:
                continue

            num_queries += 1
            num_positives.append(len(positive))
            num_negatives.append(len(negative))

            model_input = [[query, doc] for doc in docs]
            pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
            
            scores_list.extend(list(pred_scores))
            
            #### elements with hits one hit is a dict : {"corpus_id": corpus_text, "score": score}
            #### corpus_id replace by corpus text
            queries_result_list = list(map(lambda idx: {"corpus_id": docs[idx], "score": pred_scores[idx]}, range(len(docs))))
            score_calculator_ext = map_dev_samples_to_score_calculator_format({0: instance})
            score_dict = score_calculator_ext.compute_metrics([queries_result_list])
            all_scores.append(score_dict["map@k"][100])
    
        mean_map = np.mean(all_scores)
        logger.info("Queries: {} \t Positives: Min {:.1f}, Mean {:.1f}, Max {:.1f} \t Negatives: Min {:.1f}, Mean {:.1f}, Max {:.1f}".format(num_queries, np.min(num_positives), np.mean(num_positives), np.max(num_positives), np.min(num_negatives), np.mean(num_negatives), np.max(num_negatives)))
        logger.info("MAP@: {:.2f}".format(mean_map*100))
        
        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)
                #writer.writerow([epoch, steps, mean_mrr])
                writer.writerow([epoch, steps, mean_map])
            json_path = os.path.join(output_path, self.score_json_file)
            output_file_exists = os.path.isfile(json_path)
            with open(json_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(["epoch", "steps", "score@"])
                writer.writerow([epoch, steps, json.dumps({"scores_list": list(map(float ,scores_list))})])
        #return mean_mrr
        return mean_map


def read_part_file(path, adjust_neg_pos_ration = None):
    json_full_paths = glob(os.path.join(path, "*.json"))
    assert len(json_full_paths) == 4
    req = {}
    for path in json_full_paths:
        val_name = path.split("/")[-1].replace(".json", "")
        with open(path, "r", encoding = "utf-8") as f:
            j_obj = json.load(f)
        if len(j_obj) == 1:
            assert type(j_obj[list(j_obj.keys())[0]]) == type([])
            j_obj = j_obj[list(j_obj.keys())[0]]
        else:
            assert val_name.endswith("_index_dict") and len(val_name.split("_")) == 3
            val_name = "_".join(np.asarray(val_name.split("_"))[[1, 0, 2]].tolist())
            j_obj = dict(map(lambda t2: (t2[1], t2[0]) ,j_obj.items()))
        req[val_name] = j_obj
    if adjust_neg_pos_ration is not None:
        assert type(adjust_neg_pos_ration) == type(0)
        neg_tuple_set = req["neg_tuple_set"]
        pos_tuple_set = req["pos_tuple_set"]
        print("ori pos {} neg {}".format(len(pos_tuple_set), len(neg_tuple_set)))
        if len(pos_tuple_set) < len(neg_tuple_set):
            if len(neg_tuple_set) / len(pos_tuple_set) >= adjust_neg_pos_ration:
                pos_num, neg_num = len(pos_tuple_set), len(pos_tuple_set) * adjust_neg_pos_ration
            else:
                pos_num, neg_num = int(len(neg_tuple_set) / adjust_neg_pos_ration), len(neg_tuple_set)
        else:
            min_size = min(map(len, [neg_tuple_set, pos_tuple_set]))
            snip_size = int(max(1, min_size / 10000))
            snip_num = int(min_size / snip_size)
            pos_num = int(min((snip_num / adjust_neg_pos_ration) * snip_size, min_size))
            neg_num = int(min((snip_num / adjust_neg_pos_ration) * snip_size * adjust_neg_pos_ration, min_size))
        neg_tuple_set = set(list(map(tuple ,neg_tuple_set))[:neg_num])
        pos_tuple_set = set(list(map(tuple ,pos_tuple_set))[:pos_num])
        req["neg_tuple_set"] = neg_tuple_set
        req["pos_tuple_set"] = pos_tuple_set
    return req

def construct_train_samples(json_obj, neg_random = False):
    train_samples = []
    label = 1
    for t2 in json_obj["pos_tuple_set"]:
        q_index, a_index = t2[0], t2[1]
        q, a = json_obj["index_question_dict"][q_index], json_obj["index_answer_dict"][a_index]
        #q = q + "<sep>"
        train_samples.append(InputExample(texts=[q, a], label=label))
    if neg_random:
        neg_len = len(json_obj["index_answer_dict"])
    label = 0
    for t2 in json_obj["neg_tuple_set"]:
        q_index, a_index = t2[0], t2[1]
        if neg_random:
            q, a = json_obj["index_question_dict"][q_index], json_obj["index_answer_dict"][np.random.randint(0, neg_len)]
        else:
            q, a = json_obj["index_question_dict"][q_index], json_obj["index_answer_dict"][a_index]
        #q = q + "<sep>"
        train_samples.append(InputExample(texts=[q, a], label=label))
    train_indexes = np.random.permutation(np.arange(len(train_samples)))
    return list(map(lambda idx: train_samples[idx], train_indexes))

def construct_dev_samples(json_obj ,num_dev_queries = int(2e3), num_max_dev_negatives = 200, neg_random = False):
    dev_samples_list = construct_train_samples(json_obj, neg_random)
    dev_samples_df = pd.DataFrame(list(map(lambda item_:(item_.texts[0], item_.texts[1], item_.label) ,dev_samples_list)), columns = ["q", "a", "l"])
    dev_q_qid_dict = dict(map(lambda t2: (t2[1], t2[0]), enumerate(dev_samples_df["q"].drop_duplicates().tolist())))
    print("dev_samples_df shape {}, dev_q_qid_dict len {}".format(dev_samples_df.shape, len(dev_q_qid_dict)))
    dev_samples = {}
    for q, qid in dev_q_qid_dict.items():
        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': q, 'positive': set(), 'negative': set()}
    for qid in dev_samples.keys():
        qid_relate_df = dev_samples_df[dev_samples_df["q"] == dev_samples[qid]["query"]]
        dev_samples[qid]["positive"] = set(qid_relate_df[qid_relate_df["l"] == 1]["a"].tolist())
        for neg_a in set(qid_relate_df[qid_relate_df["l"] == 0]["a"].tolist()):
            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].add(neg_a) 
    return dev_samples


def merge_dev_samples(dev_samples_list):
    assert len(dev_samples_list) > 1
    query_pos_set_dict = {}
    query_neg_set_dict = {}
    for dev_samples in dev_samples_list:
        for qid, item_ in dev_samples.items():
            # item_ {'query': q, 'positive': set(), 'negative': set()}
            query = item_["query"]
            positive = item_["positive"]
            negative = item_["negative"]
            if query not in query_pos_set_dict:
                query_pos_set_dict[query] = positive
            else:
                for ele in positive:
                    query_pos_set_dict[query].add(ele)
            if query not in query_neg_set_dict:
                query_neg_set_dict[query] = negative
            else:
                for ele in negative:
                    query_neg_set_dict[query].add(ele)
    assert set(query_pos_set_dict.keys()) == set(query_neg_set_dict.keys())
    merge_dev_samples = {}
    for qid, query in enumerate(query_pos_set_dict.keys()):
        merge_dev_samples[qid] =             {'query': query, 'positive': query_pos_set_dict[query], 'negative': query_neg_set_dict[query]}
    return merge_dev_samples


def merge_dev_samples_add_neg(dev_samples_list, add_num = None, after_size = 500):
    assert len(dev_samples_list) > 1
    query_pos_set_dict = {}
    query_neg_set_dict = {}
    all_negs = reduce(lambda a, b: a.union(b) ,map(lambda dev_samples: reduce(lambda a, b: a.union(b) ,map(lambda item_:set(item_["negative"]) ,dev_samples.values())), dev_samples_list))
    assert type(all_negs) == type(set([]))
    assert set(map(type, all_negs)) == set([type("")])
    all_negs = list(all_negs)
    for dev_samples in dev_samples_list:
        for qid, item_ in dev_samples.items():
            # item_ {'query': q, 'positive': set(), 'negative': set()}
            query = item_["query"]
            positive = item_["positive"]
            negative = item_["negative"]
            if query not in query_pos_set_dict:
                query_pos_set_dict[query] = positive
            else:
                for ele in positive:
                    query_pos_set_dict[query].add(ele)
            if query not in query_neg_set_dict:
                query_neg_set_dict[query] = negative
            else:
                for ele in negative:
                    query_neg_set_dict[query].add(ele)
            if add_num is not None:
                assert type(add_num) == type(0)
                all_negs_indexes = np.random.permutation(np.arange(len(all_negs)))
                all_negs_add = list(map(lambda idx: all_negs[idx], all_negs_indexes[:add_num]))
                for ele in all_negs_add:
                    if ele not in positive:
                        query_neg_set_dict[query].add(ele)
    assert set(query_pos_set_dict.keys()) == set(query_neg_set_dict.keys())
    merge_dev_samples = {}
    for qid, query in enumerate(query_pos_set_dict.keys()):
        if len(merge_dev_samples) >= after_size:
            break
        merge_dev_samples[qid] =             {'query': query, 'positive': query_pos_set_dict[query], 'negative': query_neg_set_dict[query]}
    return merge_dev_samples


train_json_obj = read_part_file("train_file_faiss_10", adjust_neg_pos_ration = None)
train_samples = construct_train_samples(train_json_obj, neg_random = True)


valid_json_obj = read_part_file("valid_file_faiss", adjust_neg_pos_ration = None)


dev_samples = construct_dev_samples(valid_json_obj, neg_random = True)



model_name = 'xlm-roberta-base'
model_save_path = 'output/training_ms-marco_cross-encoder-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model = CrossEncoder_Dict_Eval(model_name, num_labels=1, max_length=512)



train_batch_size = 10
num_epochs = 10
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)



mrr_sum_evaluator = CERerankingEvaluatorSUM(dev_samples, name='mrr-train-eval')
map_evaluator = ScoreEvaluator(dev_samples, name = "map-train-eval")
dict_evaluator = DictionaryEvaluator([mrr_sum_evaluator, map_evaluator])


warmup_steps = 1000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=dict_evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False)

#Save latest model
model.save(model_save_path+'-latest')



