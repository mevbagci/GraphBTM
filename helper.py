import json
import os
from tqdm import tqdm
from typing import Dict, Union, List


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
    vocab_to_id = {}
    counter = 0
    for c, tweet_id in enumerate(tqdm(twitter_data, desc=F"Read twitter data")):
        if limited:
            if counter == x:
                break
        if "text" in twitter_data[tweet_id]:
            if "stanza_cleaned" in twitter_data[tweet_id]:
                text = twitter_data[tweet_id]["stanza_cleaned"]
                lang = twitter_data[tweet_id]["language"]
                if len(wanted_langs) > 0:
                    if lang not in wanted_langs:
                        continue
                if len(text.split()) < min_length:
                    continue
                for word in text.split():
                    if word not in vocab_to_id:
                        vocab_to_id[word] = counter
                        counter += 1

    for c, tweet_id in enumerate(tqdm(twitter_data, desc=F"Read twitter data")):
        if limited:
            if counter == x:
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
                    text_to_id.append(vocab_to_id[word])
                text_as_ids.append(text_to_id)
                tweet_ids.append(tweet_id)
        # if len(text.split()) < min_length:
        #     continue
        # texts.append(text)
        # tweet_ids.append(tweet_id)
        # counter += 1
    return vocab_to_id, text_as_ids, tweet_ids
