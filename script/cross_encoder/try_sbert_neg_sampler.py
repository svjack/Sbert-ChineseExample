#!/usr/bin/env python
# coding: utf-8

# In[1]:
import json
import logging
import os
import pickle
import random
import time
import traceback
from functools import reduce

import faiss
import numpy as np
import pandas as pd
import scipy.spatial
import torch
from elasticsearch import Elasticsearch, helpers
from es_pandas import es_pandas
from IPython import embed
from sentence_transformers import InputExample, SentenceTransformer, util

es_host = 'localhost:9200'
train_part, test_part, valid_part = map(lambda save_type: pd.read_csv(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir, "data", "{}_part.csv".format(save_type))
).dropna(), ["train", "test", "valid"])


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
if ep.ic.exists("train_part"):
    ep.ic.delete(index = "train_part")


ep.init_es_tmpl(train_part.head(1000), "train_part_doc_type", delete=True)
valid_part_tmp = ep.es.indices.get_template("train_part_doc_type")
es_index = valid_part_tmp["train_part_doc_type"]
es_index["mappings"]["properties"]["question"] =  {
                  "type": "text",
                }
es_index["mappings"]["properties"]["answer"] =  {
                  "type": "text",
                }
es_index = {"mappings": es_index["mappings"]}
ep.es.indices.create(index='train_part', body=es_index, ignore=[400])


chunk_size = 10000
range_list = list(range(0, train_part.shape[0], chunk_size))
if train_part.shape[0] not in range_list:
    range_list.append(train_part.shape[0])
assert "".join(map(str ,range_list)).startswith("0") and "".join(map(str ,range_list)).endswith("{}".format(train_part.shape[0]))

for i in range(len(range_list) - 1):
    part_tiny = train_part.iloc[range_list[i]:range_list[i+1]]
    ep.to_es(part_tiny, "train_part")

assert reduce(lambda a, b: a + b, map(lambda df: df.shape[0] ,ep.to_pandas_iter("train_part", chunk_size = chunk_size))) == train_part.shape[0]

if ep.ic.exists("valid_part"):
    ep.ic.delete(index = "valid_part")


ep.init_es_tmpl(train_part.head(1000), "valid_part_doc_type", delete=True)
valid_part_tmp = ep.es.indices.get_template("valid_part_doc_type")
es_index = valid_part_tmp["valid_part_doc_type"]
es_index["mappings"]["properties"]["question"] =  {
                  "type": "text",
                }
es_index["mappings"]["properties"]["answer"] =  {
                  "type": "text",
                }
es_index = {"mappings": es_index["mappings"]}
ep.es.indices.create(index='valid_part', body=es_index, ignore=[400])


chunk_size = 10000
range_list = list(range(0, valid_part.shape[0], chunk_size))
if valid_part.shape[0] not in range_list:
    range_list.append(valid_part.shape[0])
assert "".join(map(str ,range_list)).startswith("0") and "".join(map(str ,range_list)).endswith("{}".format(valid_part.shape[0]))

for i in range(len(range_list) - 1):
    part_tiny = valid_part.iloc[range_list[i]:range_list[i+1]]
    ep.to_es(part_tiny, "valid_part")

assert reduce(lambda a, b: a + b, map(lambda df: df.shape[0] ,ep.to_pandas_iter("valid_part", chunk_size = chunk_size))) == valid_part.shape[0]


class SentenceBERTNegativeSampler():
    """
    Sample candidates from a list of candidates using dense embeddings from sentenceBERT.

    Args:
        candidates: list of str containing the candidates
        num_candidates_samples: int containing the number of negative samples for each query.
        embeddings_file: str containing the path to cache the embeddings.
        sample_data: int indicating amount of candidates in the index (-1 if all)
        pre_trained_model: str containing the pre-trained sentence embedding model, 
            e.g. bert-base-nli-stsb-mean-tokens.
    """
    def __init__(self, candidates, num_candidates_samples, embeddings_file, sample_data, 
                pre_trained_model='bert-base-nli-stsb-mean-tokens', seed=42):
        random.seed(seed)
        self.candidates = candidates
        self.num_candidates_samples = num_candidates_samples
        self.pre_trained_model = pre_trained_model

        #self.model = SentenceTransformer(self.pre_trained_model)
        self.model = SentenceTransformer(self.pre_trained_model, device = "cpu")
        #extract the name of the folder with the pre-trained sentence embedding        
        if os.path.isdir(self.pre_trained_model):
            self.pre_trained_model = self.pre_trained_model.split("/")[-1] 

        self.name = "SentenceBERTNS_"+self.pre_trained_model
        self.sample_data = sample_data
        self.embeddings_file = embeddings_file

        self._calculate_sentence_embeddings()
        self._build_faiss_index()

    def _calculate_sentence_embeddings(self):
        """
        Calculates sentenceBERT embeddings for all candidates.
        """
        embeds_file_path = "{}_n_sample_{}_pre_trained_model_{}".format(self.embeddings_file,
                                                                        self.sample_data,
                                                                        self.pre_trained_model)
        if not os.path.isfile(embeds_file_path):
            logging.info("Calculating embeddings for the candidates.")
            self.candidate_embeddings = self.model.encode(self.candidates, show_progress_bar=True)
            with open(embeds_file_path, 'wb') as f:
                pickle.dump(self.candidate_embeddings, f)
        else:
            with open(embeds_file_path, 'rb') as f:
                self.candidate_embeddings = pickle.load(f)
    
    def _build_faiss_index(self):
        """
        Builds the faiss indexes containing all sentence embeddings of the candidates.
        """
        self.index = faiss.IndexFlatL2(self.candidate_embeddings[0].shape[0])   # build the index
        self.index.add(np.array(self.candidate_embeddings))
        logging.info("There is a total of {} candidates.".format(len(self.candidates)))
        logging.info("There is a total of {} candidate embeddings.".format(len(self.candidate_embeddings)))
        logging.info("Faiss index has a total of {} candidates".format(self.index.ntotal))

    def sample(self, query_str, relevant_docs):
        """
        Samples from a list of candidates using dot product sentenceBERT similarity.
        
        If the samples match the relevant doc, then removes it and re-samples randomly.
        The method uses faiss index to be efficient.

        Args:
            query_str: the str of the query to be used for the dense similarity matching.
            relevant_docs: list with the str of the relevant documents, to avoid sampling them as negative sample.
            
        Returns:
            A triplet containing the list of negative samples, 
            whether the method had retrieved the relevant doc and 
            if yes its rank in the list.
        """
        query_embedding = self.model.encode([query_str], show_progress_bar=False)
        
        distances, idxs = self.index.search(np.array(query_embedding), self.num_candidates_samples)        
        sampled_initial = [self.candidates[idx] for idx in idxs[0]]
        
        was_relevant_sampled = False
        relevant_doc_rank = -1
        sampled = []
        for i, d in enumerate(sampled_initial):
            if d in relevant_docs:
                was_relevant_sampled = True
                relevant_doc_rank = i
            else:
                sampled.append(d)

        while len(sampled) != self.num_candidates_samples: 
                sampled = sampled +                     [d for d in random.sample(self.candidates, self.num_candidates_samples-len(sampled))  
                        if d not in relevant_docs]
        return sampled, was_relevant_sampled, relevant_doc_rank





#chunk_size = 10000
#train_part = pd.concat(list(ep.to_pandas_iter("train_part", chunk_size = chunk_size)), axis = 0)
candidates = train_part["answer"].tolist()


num_candidates_samples = 30
embeddings_file = os.path.join(os.path.abspath(""), "train_sbert_emb_cache")
sample_data = -1
pre_trained_model = os.path.join(os.path.abspath(""), "bi_encoder_save")
sbert_sampler = SentenceBERTNegativeSampler(candidates, num_candidates_samples, embeddings_file, sample_data,
                                           pre_trained_model)


def part_gen_constructor(sampler, part_df):
    #question_neg_dict = {}
    for question, df in part_df.groupby("question"):
        pos_answer_list = df["answer"].tolist()
        negs = sbert_sampler.sample(question, pos_answer_list)
        #negs = sbert_sampler.sample(question, [])
        #neg_mg_df = pd.merge(train_part_tiny, pd.DataFrame(np.asarray(negs[0]).reshape([-1, 1]), columns = ["answer"]), on = "answer", how = "inner")
        #question_neg_dict[question] = neg_mg_df
        for pos_answer in pos_answer_list:
            yield InputExample(texts=[question, pos_answer], label=1)
        for neg_answer in negs[0]:
            yield InputExample(texts=[question, neg_answer], label=0)


def json_save(input_collection, path):
    assert path.endswith(".json")
    assert type(input_collection) in [type({}), type(set([]))]
    with open(path, "w", encoding = "utf-8") as f:
        if type(input_collection) == type({}):
            #json.dump(input_collection, f, encoding = "utf-8")
            pass
        else:
            input_collection = {path.split("/")[-1].replace(".json", ""): list(input_collection)}
        json.dump(input_collection, f)
    print("save to {}".format(path))


def produce_question_answer_sample_in_file_format(part_gen, chunck_size = 1000, save_times = 1, sub_dir = None):
    question_index_dict = {}
    answer_index_dict = {}
    pos_tuple_set = set([])
    neg_tuple_set = set([])
    have_save = 0
    #for idx, item_ in enumerate(part_gen):
    idx = 0
    while True:
        item_ = part_gen.__next__()
        idx += 1
        question, answer = item_.texts
        if question not in question_index_dict:
            question_index_dict[question] = len(question_index_dict)
        if answer not in answer_index_dict:
            answer_index_dict[answer] = len(answer_index_dict)    
        label = item_.label
        assert label in [0, 1]
        if label == 1:
            pos_tuple_set.add((question_index_dict[question], answer_index_dict[answer]))
        else:
            neg_tuple_set.add((question_index_dict[question], answer_index_dict[answer]))
        if sub_dir is not None and not os.path.exists(os.path.join(os.path.abspath(""), sub_dir)):
            assert type(sub_dir) == type("") and "/" not in sub_dir
            os.mkdir(os.path.join(os.path.abspath(""), sub_dir))
        if (idx + 1) % chunck_size == 0:
            for c in ["question_index_dict", "answer_index_dict", "pos_tuple_set", "neg_tuple_set"]:
                if sub_dir is None:
                    exec("json_save({}, '{}.json')".format(c, os.path.join(os.path.abspath(""), c)))
                else:
                    exec("json_save({}, '{}.json')".format(c, os.path.join(os.path.abspath(""), sub_dir, c)))
            have_save += 1
            print("have_save in {} step".format(idx + 1))
            if have_save >= save_times:
                return


train_part_gen = part_gen_constructor(sbert_sampler, train_part)
produce_question_answer_sample_in_file_format(train_part_gen, chunck_size = 10000, save_times = 10000,
                                             sub_dir = "train_file_faiss_10")




