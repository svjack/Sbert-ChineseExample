#!/usr/bin/env python
# coding: utf-8
import gzip
import logging
import math
import os
import random
import tarfile
from collections import defaultdict
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
from sentence_transformers import (LoggingHandler, SentenceTransformer,
                                   evaluation, losses, models, util)
from torch.utils.data import DataLoader, Dataset, IterableDataset

#train_part, test_part, valid_part = map(lambda save_type: pd.read_csv(os.path.join(os.path.abspath(""), "{}_part.csv".format(save_type))).dropna(), ["train", "test", "valid"])
train_part, test_part, valid_part = map(lambda save_type: pd.read_csv(os.path.join("..data/", "{}_part.csv".format(save_type))).dropna(), ["train", "test", "valid"])

from sentence_transformers import InputExample
class TripletsDataset(Dataset):
    def __init__(self, model, qa_df):
        assert set(["question", "answer", "q_idx"]).intersection(set(qa_df.columns.tolist())) == set(["question", "answer", "q_idx"])
        self.model = model
        self.qa_df = qa_df
        self.q_idx_set = set(qa_df["q_idx"].value_counts().index.tolist())

    def __getitem__(self, index):
        #raise NotImplementedError
        label = torch.tensor(1, dtype=torch.long)
        choice_s = self.qa_df.iloc[index]
        query_text, pos_text, q_idx = choice_s.loc["question"], choice_s.loc["answer"],  choice_s.loc["q_idx"]
        query_text, pos_text, q_idx = choice_s.loc["question"], choice_s.loc["answer"],  choice_s.loc["q_idx"]
        neg_q_idx = np.random.choice(list(self.q_idx_set.difference(set([q_idx]))))
        neg_text = self.qa_df[self.qa_df["q_idx"] == neg_q_idx].sample()["answer"].iloc[0]
        ####  InputExample(texts=['I can\'t log in to my account.',
        #'Unable to access my account.',
        #'I need help with the payment process.'],
        #label=1),
        return InputExample(texts = [query_text, pos_text, neg_text], label = 1)
        '''
        return [self.model.tokenize(query_text),
                self.model.tokenize(pos_text),
                self.model.tokenize(neg_text)], label
        '''
        #return (query_text, pos_text, q_idx)

    def __len__(self):
        return self.qa_df.shape[0]


class NoSameLabelsBatchSampler:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.idx_org = list(range(len(dataset)))
        random.shuffle(self.idx_org)
        self.idx_copy = self.idx_org.copy()
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        labels = set()
        num_miss = 0

        num_batches_returned = 0
        while num_batches_returned < self.__len__():
            if len(self.idx_copy) == 0:
                random.shuffle(self.idx_org)
                self.idx_copy = self.idx_org.copy()

            idx = self.idx_copy.pop(0)
            #label = self.dataset[idx][1].cpu().tolist()
            label = self.dataset.qa_df["q_idx"].iloc[idx]
            
            if label not in labels:
                num_miss = 0
                batch.append(idx)
                labels.add(label)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    labels = set()
                    num_batches_returned += 1
            else:
                num_miss += 1
                self.idx_copy.append(idx) #Add item again to the end

                if num_miss >= len(self.idx_copy): #To many failures, flush idx_copy and start with clean
                    self.idx_copy = []

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def transform_part_df_into_Evaluator_format(part_df):
    req = part_df.copy()
    req["qid"] = req["question"].fillna("").map(hash).map(str)
    req["cid"] = req["answer"].fillna("").map(hash).map(str)
    queries = dict(map(tuple ,req[["qid", "question"]].drop_duplicates().values.tolist()))
    corpus = dict(map(tuple ,req[["cid", "answer"]].drop_duplicates().values.tolist()))
    qid_cid_set_df = req[["qid", "cid"]].groupby("qid")["cid"].apply(set).apply(sorted).apply(tuple).reset_index()
    qid_cid_set_df.columns = ["qid", "cid_set"]
    relevant_docs = dict(map(tuple ,qid_cid_set_df.drop_duplicates().values.tolist()))
    relevant_docs = dict(map(lambda t2: (t2[0], set(t2[1])) ,relevant_docs.items()))
    return queries, corpus,  relevant_docs


dev_queries, dev_corpus, dev_rel_docs = transform_part_df_into_Evaluator_format(valid_part.sample(frac=0.1))
ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs, name='ms-marco-train_eval', batch_size=2)



model_str = "xlm-roberta-base"
#word_embedding_model = models.Transformer(model_str, max_seq_length=512)
word_embedding_model = models.Transformer(model_str, max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])



train_dataset = TripletsDataset(model=model, qa_df = train_part.sample(frac = 1.0, replace=False))
bs_obj = NoSameLabelsBatchSampler(train_dataset, batch_size=8)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, batch_sampler=bs_obj, num_workers=1)
train_loss = losses.MultipleNegativesRankingLoss(model=model)


model_save_path = os.path.join(os.path.abspath(""), "bi_encoder_save")
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)


model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=ir_evaluator,
          epochs=10,
          warmup_steps=1000,
          output_path=model_save_path,
          evaluation_steps=5000,
          use_amp=True
          )









