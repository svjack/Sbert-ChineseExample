#!/usr/bin/env python
# coding: utf-8
import gzip
import json
import logging
import os
import tarfile
import time
from datetime import datetime
from functools import partial, reduce
from glob import glob
from typing import Callable, Dict, List, Type

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from elasticsearch import Elasticsearch, helpers
from es_pandas import es_pandas
from sentence_transformers import (InputExample, LoggingHandler,
                                   SentenceTransformer, util)
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

pd.set_option("display.max_rows", 200)
es_host = 'localhost:9200'


bi_model_path = os.path.join(os.path.dirname("__file__"), os.path.pardir, "bi_encoder_save/")
bi_model = SentenceTransformer(bi_model_path, device = "cpu")


cross_model_path = "output/training_ms-marco_cross-encoder-xlm-roberta-base-2021-01-17_14-43-23_map-train-eval"
cross_model = CrossEncoder(cross_model_path, num_labels=1, max_length=512, device = "cpu")


class es_pandas_edit(es_pandas):
    @staticmethod
    def serialize(row, columns, use_pandas_json, iso_dates):
        if use_pandas_json:
            return json.dumps(dict(zip(columns, row)), iso_dates=iso_dates)
        return dict(zip(columns, [None if (all(pd.isna(r)) if (hasattr(r, "__len__") and type(r) != type("")) else pd.isna(r)) else r for r in row]))
    def to_pandas_iter(self, index, query_rule=None, heads=[], dtype={}, infer_dtype=False, show_progress=True, 
                  chunk_size = None, **kwargs):
        if query_rule is None:
            query_rule = {'query': {'match_all': {}}}
        count = self.es.count(index=index, body=query_rule)['count']
        if count < 1:
            raise Exception('Empty for %s' % index)
        query_rule['_source'] = heads
        anl = helpers.scan(self.es, query=query_rule, index=index, **kwargs)
        source_iter = self.get_source(anl, show_progress = show_progress, count = count)
        print(source_iter)
        if chunk_size is None:
            df = pd.DataFrame(list(source_iter)).set_index('_id')
            if infer_dtype:
                dtype = self.infer_dtype(index, df.columns.values)
                if len(dtype):
                    df = df.astype(dtype)
            yield df
            return
        assert type(chunk_size) == type(0)
        def map_list_of_dicts_into_df(list_of_dicts, set_index = "_id"):
            from collections import defaultdict
            req = defaultdict(list)
            for dict_ in list_of_dicts:
                for k, v in dict_.items():
                    req[k].append(v)
            req = pd.DataFrame.from_dict(req)
            if set_index:
                assert set_index in req.columns.tolist()
            t_df = req.set_index(set_index)
            if infer_dtype:
                dtype = self.infer_dtype(index, t_df.columns.values)
                if len(dtype):
                    t_df = t_df.astype(dtype)
            return t_df
        list_of_dicts = []
        for dict_ in source_iter:
            list_of_dicts.append(dict_)
            if len(list_of_dicts) >= chunk_size:
                yield map_list_of_dicts_into_df(list_of_dicts)
                list_of_dicts = []
        if list_of_dicts:
            yield map_list_of_dicts_into_df(list_of_dicts)


ep = es_pandas_edit(es_host)
ep.ic.get_alias("*")

chunk_size = 1000
valid_part_from_es_iter = ep.to_pandas_iter(index = "valid_part", chunk_size = chunk_size)


valid_part_tiny = None
for ele in valid_part_from_es_iter:
    valid_part_tiny = ele
    break
del valid_part_from_es_iter


if ep.ic.exists("valid_part_tiny"):
    ep.ic.delete(index = "valid_part_tiny")


ep.init_es_tmpl(valid_part_tiny, "valid_part_tiny_doc_type", delete=True)
valid_part_tmp = ep.es.indices.get_template("valid_part_tiny_doc_type")


es_index = valid_part_tmp["valid_part_tiny_doc_type"]
es_index["mappings"]["properties"]["question_emb"] =  {
                  "type": "dense_vector",
                  "dims": 768
                }
es_index["mappings"]["properties"]["answer_emb"] =  {
                  "type": "dense_vector",
                  "dims": 768
                }
es_index["mappings"]["properties"]["question"] =  {
                  "type": "text",
                }
es_index["mappings"]["properties"]["answer"] =  {
                  "type": "text",
                }
es_index = {"mappings": es_index["mappings"]}


ep.es.indices.create(index='valid_part_tiny', body=es_index, ignore=[400])
question_embeddings = bi_model.encode(valid_part_tiny["question"].tolist(), convert_to_tensor=True, show_progress_bar=True)
answer_embeddings = bi_model.encode(valid_part_tiny["answer"].tolist(), convert_to_tensor=True, show_progress_bar=True)

valid_part_tiny["question_emb"] = question_embeddings.cpu().numpy().tolist()
valid_part_tiny["answer_emb"] = answer_embeddings.cpu().numpy().tolist()

ep.to_es(valid_part_tiny, "valid_part_tiny")

chunk_size = 1000
valid_part_tiny = list(ep.to_pandas_iter(index = "valid_part_tiny", chunk_size = None))[0]


def search_by_embedding_in_es(index = "valid_part" ,embedding = np.asarray(valid_part_tiny["question_emb"].iloc[0]), on_column = "answer_emb"):
    vector_search_one = ep.es.search(index=index, body={
          "query": {
            "script_score": {
              "query": {
                "match_all": {}
              },
              "script": {
                "source": "cosineSimilarity(params.queryVector, doc['{}']) + 1.0".format(on_column),
                "params": {
                  "queryVector": embedding
                }
              }
            }
          }
        }, ignore = [400])
    req = list(map(lambda x: (x["_source"]["question"], x["_source"]["answer"], x["_score"]) ,vector_search_one["hits"]["hits"]))
    req_df = pd.DataFrame(req, columns = ["question", "answer", "score"])
    return req_df


def search_by_text_in_es(index = "valid_part" ,text = valid_part_tiny["question"].iloc[0], on_column = "answer",
                        analyzer = "smartcn"):
    if analyzer is not None:
        bm25 = es.search(index = index,
          body={"query": 
                {
                    "match": {on_column:{"query" :text, "analyzer": analyzer} },
                    
                }
               },
         )
    else:
        bm25 = ep.es.search(index=index, body={"query": {"match": {on_column: text}}})
    req = list(map(lambda x: (x["_source"]["question"], x["_source"]["answer"], x["_score"]) ,bm25["hits"]["hits"]))
    req_df = pd.DataFrame(req, columns = ["question", "answer", "score"])
    return req_df


def valid_two_model(cross_model, ep, index, question, question_embedding, on_column = "answer_emb", size = 10):
    def search_by_embedding(ep ,index = "valid_part" ,embedding = np.asarray(valid_part_tiny["question_emb"].iloc[0]), on_column = "answer_emb"):
        vector_search_one = ep.es.search(index=index, body={
        "size": size,
          "query": {
            "script_score": {
              "query": {
                "match_all": {}
              },
              "script": {
                "source": "cosineSimilarity(params.queryVector, doc['{}']) + 1.0".format(on_column),
                "params": {
                  "queryVector": embedding
                }
              }
            }
          }
        }, ignore = [400])
        req = list(map(lambda x: (x["_source"]["question"], x["_source"]["answer"], x["_score"]) ,vector_search_one["hits"]["hits"]))
        req_df = pd.DataFrame(req, columns = ["question", "answer", "score"])
        return req_df
    search_by_emb = search_by_embedding(ep ,index = index, embedding = question_embedding, on_column = on_column)
    print("question : {}".format(question))
    preds = cross_model.predict(search_by_emb.apply(lambda r: [question, r["answer"]], axis = 1).tolist())
    search_by_emb["cross_score"] = preds.tolist()
    return search_by_emb
def produce_df(question, size = 10):
    question, question_embedding = valid_part_tiny[valid_part_tiny["question"] == question].iloc[0][["question", "question_emb"]]
    valid_df = valid_two_model(cross_model, ep, index = "valid_part_tiny", question = question, question_embedding = question_embedding, size = size)
    return valid_df


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
                        #break

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

def map_valid_df_to_score_calculator_format(query ,valid_df):
    queries_ids = [query]
    relevant_docs = {query: valid_df[valid_df["question"] == query]["answer"].tolist()}
    return ScoreCalculator(queries_ids, relevant_docs)


def df_to_mrr_score(df, query, score_col, mrr_at_k = 10):
    #model_input = [[query, doc] for doc in docs]
    #pred_scores = model.predict(model_input, convert_to_numpy=True, show_progress_bar=False)
    is_relevant = list(map(lambda t2: True if t2[1]["question"] == query else False, df.iterrows()))    
    pred_scores = df[score_col].values
    pred_scores_argsort = np.argsort(-pred_scores)  #Sort in decreasing order
    mrr_score = 0
    for rank, index in enumerate(pred_scores_argsort[0:mrr_at_k]):
        if is_relevant[index]:
            mrr_score = 1 / (rank+1)
            #mrr_score += 1 / (rank+1)
            break
    return mrr_score


question_list = valid_part_tiny["question"].value_counts().index.tolist()
valid_df = produce_df(question_list[10], size = 100)

def produce_score_dict(query ,valid_df, column = "score"):
    queries_result_list = valid_df[["answer", column]].apply(lambda x: {"corpus_id": x["answer"], "score": x[column]}, axis = 1).tolist()
    score_dict = map_valid_df_to_score_calculator_format(query, valid_df).compute_metrics([queries_result_list])
    return score_dict

produce_score_dict(question_list[10] ,valid_df, "score")
produce_score_dict(question_list[10] ,valid_df, "cross_score")
produce_score_dict(question_list[10] ,valid_df.head(20), "score")
produce_score_dict(question_list[10] ,valid_df.head(20), "cross_score")

valid_df.head(20) 
valid_df.head(20).sort_values(by = "cross_score", ascending = False)
valid_df.sort_values(by = "cross_score", ascending = False).head(10)

sns.distplot(valid_df["cross_score"])


