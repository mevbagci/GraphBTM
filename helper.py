import json
import os
from tqdm import tqdm
from typing import Dict, Union, List
import pickle
import numpy as np


def load_json(dir_json: str):
    with open(dir_json, "r", encoding="UTF-8") as json_file:
        return json.load(json_file)


def write_json(dir_json: str, json_data: str):
    os.makedirs(os.path.dirname(dir_json), exist_ok=True)
    with open(dir_json, "w", encoding="UTF-8") as json_file:
        return json.dump(json_data, json_file, indent=2)


def load_documents(twitter_data: Dict[str, Union[str, Dict[str, str]]], x: int, limited: bool = True, wanted_langs: List[str] = [], min_length: int = 0):
    tweet_ids = []
    text_as_ids = []
    corpus = ""
    vocab_to_id = {}
    counter = 0
    counter_text = 0
    for c, tweet_id in enumerate(tqdm(twitter_data, desc=F"Read twitter data")):
        if limited:
            if counter_text == x:
                break
        if "text" in twitter_data[tweet_id]:
            if "stanza_cleaned" in twitter_data[tweet_id]:
                text = twitter_data[tweet_id]["stanza_cleaned"].strip()
                lang = twitter_data[tweet_id]["language"]
                if len(wanted_langs) > 0:
                    if lang not in wanted_langs:
                        continue
                if len(text.split()) < min_length:
                    continue
                corpus += f"{text}\n"
                counter_text += 1
                for word in text.split():
                    # word = word.lower()
                    if word not in vocab_to_id:
                        vocab_to_id[word] = counter
                        counter += 1

    counter_text = 0
    for c, tweet_id in enumerate(tqdm(twitter_data, desc=F"Read twitter data")):
        if limited:
            if counter_text == x:
                break
        if "text" in twitter_data[tweet_id]:
            if "stanza_cleaned" in twitter_data[tweet_id]:
                text = twitter_data[tweet_id]["stanza_cleaned"]
                lang = twitter_data[tweet_id]["language"]
                text_to_id = []
                if len(wanted_langs) > 0:
                    if lang not in wanted_langs:
                        continue
                if len(text.split()) < min_length:
                    continue
                for word in text.split():
                    # word = word.lower()
                    text_to_id.append(vocab_to_id[word])
                text_to_id = np.array(text_to_id)
                text_as_ids.append(text_to_id)
                tweet_ids.append(tweet_id)
                counter_text += 1
        # if len(text.split()) < min_length:
        #     continue
        # texts.append(text)
        # tweet_ids.append(tweet_id)
        # counter += 1
    return corpus, vocab_to_id, text_as_ids, tweet_ids


if __name__ == '__main__':
    tweet_dir = f"/mnt/corpora2/projects/bagci/ACLTwitter/test/MeTwo_infos/MeTwo/Metwo.json"
    with open(tweet_dir, "r", encoding="UTF-8") as json_file:
        tweet = json.load(json_file)
    corpus_text, vocabs, text_id, tweet_ids = load_documents(tweet, 2000, True, min_length=5)
    base_out = f"/mnt/corpora2/projects/bagci/ACLTwitter/test/models/GraphBTM/MeTwo/data"
    os.makedirs(base_out, exist_ok=True)
    with open(f"{base_out}/corpus.txt", "w", encoding="UTF-8") as txt_file:
        txt_file.write(corpus_text)
    with open(f"{base_out}/vocab.pkl", 'wb') as f:
        pickle.dump(vocabs, f)
    np_text = np.array(text_id)
    np.save(f"{base_out}/train.txt.npy", np_text)
    with open(f"{base_out}/tweets_ods.json", "w", encoding="UTF-8") as json_file:
        json.dump(tweet_ids, json_file, indent=2)

