import pandas as pd
import json
from uuid import uuid4
from datasets import Dataset


def normalize_json(data, record_path, meta, add_ids=None):
    df = pd.json_normalize(data, record_path=record_path, meta=meta)
    if add_ids:
        df[add_ids] = [str(uuid4()) for _ in range(len(df))]
    dataset = Dataset.from_pandas(df)
    return dataset


def preprocess_data(data_file):
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    data = data["data"]
    """for d in data:
        for p in d["paragraphs"]:
            if len(p["qas"]) == 0:
                p["qas"].append({"answer": None, "question": None, "generative_answer": None, "is_impossible": None,
                                 "answer_span": None})"""
    dataset = normalize_json(normalize_json(data, "paragraphs", ["id"]), "qas", ["id", "context"], "example_id")
    return dataset

def load_dataset(train_file, val_file=None, test_file=None, seed=None):
    dataset = {}
    train = preprocess_data(train_file)
    if val_file is None and test_file is None:
        train.shuffle(seed=seed)
        train_val = train.train_test_split(test_size=0.2)
        train = train_val["train"]
        val_test = train_val["test"].train_test_split(test_size=0.5)
        val = val_test["train"]
        test = val_test["test"]
    else:
        val = preprocess_data(val_file)
        test = preprocess_data(test_file)

    dataset["train"] = train
    dataset["val"] = val
    dataset["test"] = test
    return dataset

def select_samples(dataset, num_samples=None):
    if num_samples is not None:
        max_num_samples = min(len(dataset), num_samples)
        dataset = dataset.select(range(max_num_samples))
    return dataset
