#!/usr/bin/env python
# coding: utf-8
import os
from copy import deepcopy
from functools import reduce
from glob import glob

import editdistance
import numpy as np
import pandas as pd

###https://github.com/brightmart/nlp_chinese_corpus
###https://github.com/brightmart/nlp_chinese_corpus#4%E7%A4%BE%E5%8C%BA%E9%97%AE%E7%AD%94json%E7%89%88webtext2019zh-%E5%A4%A7%E8%A7%84%E6%A8%A1%E9%AB%98%E8%B4%A8%E9%87%8F%E6%95%B0%E6%8D%AE%E9%9B%86
###https://drive.google.com/open?id=1u2yW_XohbYL2YAK6Bzc5XrngHstQTf0v

data_dir = r"/home/svjack/temp_dir/webtext2019zh"
json_files = glob(os.path.join(data_dir, "*.json"))
train_json = list(filter(lambda path: "train" in path.lower(), json_files))[0]
def json_reader(path, chunksize = 100):
    assert path.endswith(".json")
    return pd.read_json(path, lines = True, chunksize=chunksize)

train_reader = json_reader(train_json, chunksize=10000)
times = 100
df_list = []
for i, df in enumerate(train_reader):
    df_list.append(df)
    if i + 1 >= times:
        break
    
train_head_df = pd.concat(df_list, axis = 0)
content_len_df = pd.concat([train_head_df["content"], train_head_df["content"].map(len)], axis = 1)
content_len_df.columns = ["content", "c_len"]


qa_df = train_head_df[["title", "content"]].copy()
qa_df = qa_df.rename(columns = {"title": "question", "content": "answer"}).fillna("")

qa_df = qa_df[qa_df["question"].map(len) <= 500]
qa_df = qa_df[qa_df["answer"].map(len) <= 500]


quests = deepcopy(qa_df["question"])
question_cmp = pd.concat([quests.sort_values().shift(1), quests.sort_values()], axis = 1)
question_cmp["edit_val"] = question_cmp.fillna("").apply(lambda s: editdistance.eval(s.iloc[0], s.iloc[1]) / (len(s.iloc[0]) + len(s.iloc[1])), axis = 1)
question_cmp.columns = ["q0", "q1", "edit_val"]

threshold = 0.2
question_nest_list = [[]]
for idx ,r in question_cmp.iterrows():
    q0, q1, v = r.iloc[0], r.iloc[1], r.iloc[2]
    if v < threshold:
        question_nest_list[-1].append(q0)
        question_nest_list[-1].append(q1)
    else:
        question_nest_list.append([])


idx_question_df_zip = pd.DataFrame(list(map(lambda x: [x] ,question_nest_list)))

idx_question_df_zip = idx_question_df_zip[idx_question_df_zip.iloc[:, 0].map(len) > 0]
idx_question_df_zip.columns = ["question"]
idx_question_df_zip["q_idx"] = np.arange(idx_question_df_zip.shape[0]).tolist()

idx_question_df = idx_question_df_zip.explode("question")

#idx_question_df = pd.DataFrame(reduce(lambda a, b: a + b, map(lambda idx: list(map(lambda q: (idx, q), question_nest_list[idx])), range(len(question_nest_list)))))
#idx_question_df.columns = ["q_idx", "question"]
#idx_question_df.drop_duplicates().to_csv(os.path.join("/home/svjack/temp_dir/", "idx_question_df.csv"), index = False)

idx_question_df_dd =  idx_question_df.drop_duplicates()



qa_df_dd = qa_df.drop_duplicates()
cat_qa_df_with_idx = pd.merge(qa_df_dd, idx_question_df_dd, on = "question", how = "inner")
q_idx_set = set(cat_qa_df_with_idx["q_idx"].value_counts().index.tolist())

q_idx_size_bigger_or_eql_3 = ((cat_qa_df_with_idx["q_idx"].value_counts() >= 3).reset_index()).groupby("q_idx")["index"].apply(set).apply(list)[True]
q_idx_size_bigger_or_eql_3_df = cat_qa_df_with_idx[cat_qa_df_with_idx["q_idx"].isin(q_idx_size_bigger_or_eql_3)].copy()


def produce_label_list(length = 10, p_list = [0.1, 0.1, 0.8]):
    from functools import reduce
    assert sum(p_list) == 1
    p_array = np.asarray(p_list)
    assert all((p_array[:-1] <= p_array[1:]).astype(bool).tolist())
    num_array = (p_array * length).astype(np.int32)
    num_list = num_array.tolist()
    num_list = list(map(lambda x: max(x, 1), num_list))
    num_list[-1] = length - sum(num_list[:-1])
    return np.random.permutation(reduce(lambda a, b: a + b ,map(lambda idx: [idx] * num_list[idx], range(len(p_list)))))

q_idx_size_bigger_or_eql_3_df["r_idx"] = q_idx_size_bigger_or_eql_3_df.index.tolist()

def map_r_idx_list_to_split_label_zip(r_idx_list):
    split_label_list = produce_label_list(len(r_idx_list))
    assert len(split_label_list) == len(r_idx_list)
    return zip(*[r_idx_list, split_label_list])

r_idx_split_label_items = reduce(lambda a, b: a + b ,q_idx_size_bigger_or_eql_3_df.groupby("q_idx")["r_idx"].apply(set).apply(list).apply(map_r_idx_list_to_split_label_zip).apply(list).tolist())
r_idx_split_label_df = pd.DataFrame(r_idx_split_label_items)
r_idx_split_label_df.columns = ["r_idx", "split_label"]
assert r_idx_split_label_df.shape[0] == pd.merge(q_idx_size_bigger_or_eql_3_df, r_idx_split_label_df, on = "r_idx", how = "inner").shape[0]

q_idx_size_bigger_or_eql_3_df_before_split = pd.merge(q_idx_size_bigger_or_eql_3_df, r_idx_split_label_df, on = "r_idx", how = "inner")
train_part = q_idx_size_bigger_or_eql_3_df_before_split[q_idx_size_bigger_or_eql_3_df_before_split["split_label"] == 2].copy()
train_part = pd.concat([train_part, cat_qa_df_with_idx[(1 - cat_qa_df_with_idx["q_idx"].isin(q_idx_size_bigger_or_eql_3)).astype(bool)].copy()], axis = 0)
valid_part = q_idx_size_bigger_or_eql_3_df_before_split[q_idx_size_bigger_or_eql_3_df_before_split["split_label"] == 0].copy()
test_part = q_idx_size_bigger_or_eql_3_df_before_split[q_idx_size_bigger_or_eql_3_df_before_split["split_label"] == 1].copy()

assert set(valid_part["q_idx"].tolist()) == set(test_part["q_idx"].tolist())
assert set(valid_part["q_idx"].tolist()) == set(valid_part["q_idx"].tolist()).intersection(train_part["q_idx"].tolist())

train_part.to_csv(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir, "data", "train_part.csv"), index = False)
test_part.to_csv(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir, "data", "test_part.csv"), index = False)
valid_part.to_csv(os.path.join(os.path.dirname("__file__"), os.path.pardir, os.path.pardir, "data", "valid_part.csv"), index = False)
