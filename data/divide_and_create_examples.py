import json 
import argparse
from collections import defaultdict
import os
import pickle
import gzip
import re

import pandas as pd
from tqdm import tqdm
import numpy as np

from ._tokenizer import Vocab, Indexlizer
from ._stop_words import ENGLISH_STOP_WORDS


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='/raid/hanszeng/datasets/amazon_dataset/reviews_Toys_and_Games_5.json.gz')
    parser.add_argument("--dest_dir", default="./Toys_and_Games_5/randomly_rel_match/")
    parser.add_argument("--rv_num_keep_prob", default=0.9, type=float)
    parser.add_argument("--max_rv_len", default=60, type=int)

    args = parser.parse_args() 

    return args 

def truncate_pad_tokens(tokens, max_seq_len, pad_token):
    # truncate 
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    # pad
    res_length = max_seq_len - len(tokens)
    tokens = tokens + [pad_token] * res_length
    return tokens

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def split_data(args):
    path = args.data_path
    dest_dir = args.dest_dir

    f = gzip.open(path)

    users = [] 
    items = [] 
    ratings = [] 
    reviews = [] 
    times = []

    for line in f:
        js_dict = json.loads(line)
        if str(js_dict['reviewerID'])=='unknown':
            print("unknown user")
            continue
        if str(js_dict['asin'])=='unknown':
            print("unknown item")
            continue

        users.append(js_dict["reviewerID"])
        items.append(js_dict["asin"])
        ratings.append(js_dict["overall"])
        reviews.append(js_dict["reviewText"])
        times.append(js_dict["unixReviewTime"])

    df = pd.DataFrame({"user_id": pd.Series(users),
                        "item_id": pd.Series(items),
                        "rating": pd.Series(ratings),
                        "review": pd.Series(reviews),
                        "time": pd.Series(times)})

    # numerize user and item
    users = list(df["user_id"])
    items = list(df["item_id"])
    user2id = {u:i for i, u in enumerate(np.unique(users))}
    item2id = {it:i for i, it in enumerate(np.unique(items))}
    df["user_id"] = df["user_id"].apply(lambda x: user2id[x])
    df["item_id"] = df["item_id"].apply(lambda x: item2id[x])

    del users
    del items
    del ratings
    del reviews
    del times

    # sort df by `user_id` and `time` 
    df = df.sort_values(by=["user_id", "time"]).reset_index(drop=True)

    print(df.iloc[:10])
    print("number of user: {}, item: {}".format(len(user2id), len(item2id)))

    # randomly divide train, validation, test set by 0.8, 0.1, 0.1.
    np.random.seed(20200616)
    num_samples = len(df)
    train_idx = np.random.choice(num_samples, int(num_samples*0.8), replace=False)
    remain_idx = list(set(range(num_samples)) - set(train_idx))
    train_idx = list(train_idx)
    num_remain = len(remain_idx)
    valid_idx = remain_idx[:int(num_remain * 0.5)]
    test_idx = remain_idx[int(num_remain * 0.5):]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(train_df.head())
    print("sample of total: {}".format(len(df)))
    print("samples of train_df: {}, valid_df: {}, test_df: {}".format(len(train_df), len(valid_df), len(test_df)))

    write_pickle(os.path.join(args.dest_dir, "raw_df.pkl"), df)

    return train_df, valid_df, test_df



def create_meta(df, args):
    meta = {}
    # statistics
    reviews = list(df.review)
    indexlizer = Indexlizer(reviews, special_tokens=["<pad>", "<unk>"], preprocessor=clean_str, stop_words=ENGLISH_STOP_WORDS, 
                        max_len=args.max_rv_len)
    indexlized_reviews = indexlizer.transform(reviews)
    df["idxed_review"] = indexlized_reviews
    print("review length: 0.5, 0.7, 0.9, 0.95: {}".format(np.quantile(indexlizer.review_lengths, [0.5, 0.7, 0.9, 0.95])))

    ur_nums = np.array(df.groupby("user_id")["review"].agg(["count"]))
    ir_nums = np.array(df.groupby("item_id")["review"].agg(["count"]))
    ur_num = np.quantile(ur_nums, args.rv_num_keep_prob)
    ir_num = np.quantile(ir_nums, args.rv_num_keep_prob)
    print(f"review num for user, item at {args.rv_num_keep_prob} quantile: is {ur_num}, {ir_num}" )

    meta["ur_num"] = int(ur_num)
    meta["ir_num"] = int(ir_num)
    meta["rv_num"] = int(ur_num)
    meta["rv_len"] = args.max_rv_len
    meta["user_num"] = df.user_id.max() + 1
    meta["item_num"] = df.item_id.max() + 1 # 加上 pad_idx 0, 并且考虑了空袭
    print(df.user_id.max(), df.item_id.max())

    # create meta for reviews and rids
    rv_num = meta["rv_num"]

    user_reviews = defaultdict(list)
    item_reviews = defaultdict(list)
    user_rids = defaultdict(list)
    item_rids = defaultdict(list)

    train_users = list(df["user_id"])
    train_items = list(df["item_id"])
    train_reviews = list(df["idxed_review"])

    for user, item, review in zip(train_users, train_items, train_reviews):
        user_reviews[user].append(review)
        item_reviews[item].append(review)
        user_rids[user].append(item)
        item_rids[item].append(user)

    # truncate or pad 
    for user, list_of_review in tqdm(user_reviews.items()):
        padded_review = [0] * args.max_rv_len
        user_reviews[user] = truncate_pad_tokens(list_of_review, rv_num, padded_review)

    for user, rids in tqdm(user_rids.items()):
        user_rids[user] = truncate_pad_tokens(rids, rv_num, 0)
    
    for item, list_of_review in tqdm(item_reviews.items()):
        padded_review = [0] * args.max_rv_len
        item_reviews[item] = truncate_pad_tokens(list_of_review, rv_num, padded_review)
    
    for item, rids in tqdm(item_rids.items()):
        item_rids[item] = truncate_pad_tokens(rids, rv_num, 0)

    meta["user_reviews"] = user_reviews
    meta["user_rids"] = user_rids
    meta["item_reviews"] = item_reviews
    meta["item_rids"] = item_rids
    meta["indexlizer"] = indexlizer

    # test 
    t_uid, t_iid = 1, 45 
    t_reviews = meta["user_reviews"][t_uid][1:3]
    print("uid: ", t_uid)
    print("decoded review: ",  list(map(indexlizer.transform_idxed_review, t_reviews)))
    print(len(meta["user_reviews"][t_uid]), len(meta["user_rids"][t_uid]))

    t_reviews = meta["user_reviews"][t_iid][1:3]
    print("iid: ", t_iid)
    print("decoded review: ",  list(map(indexlizer.transform_idxed_review, t_reviews)))
    print(len(meta["item_reviews"][t_iid]), len(meta["item_rids"][t_iid]))

    return meta 


def create_examples(df, meta, set_name):
    users = list(df.user_id)
    items = list(df.item_id)
    ratings = list(df.rating)

    examples = list(zip(users, items, ratings))

    if set_name == "train":
        return examples 
    else:
        u_reviews = meta["user_reviews"]
        i_reviews = meta["item_reviews"]
        print(len(u_reviews), len(i_reviews))
        print(list(u_reviews.keys())[:100])
        filterd_examples = []
        for example in tqdm(examples):
            user, item = example[0], example[1]
            if user in u_reviews and item in i_reviews:
                filterd_examples.append(example)
            else:
                print("{} example {} is not valid".format(set_name, example))
        print("the example being removed: ", len(examples)-len(filterd_examples))

        return examples


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    train_df, valid_df, test_df = split_data(args)
    meta = create_meta(train_df, args)

    train_examples = create_examples(train_df, None, "train")
    valid_examples = create_examples(valid_df, meta, "valid")
    test_examples = create_examples(test_df, meta, "test")

    # print meta 
    for k, v in meta.items():
        if isinstance(v, dict):
            print(k)
        else:
            print(k, v)

    write_pickle(os.path.join(args.dest_dir, "meta.pkl"), meta)
    write_pickle(os.path.join(args.dest_dir, "train_exmaples.pkl"), train_examples)
    write_pickle(os.path.join(args.dest_dir, "valid_exmaples.pkl"), valid_examples)
    write_pickle(os.path.join(args.dest_dir, "test_exmaples.pkl"), test_examples)
