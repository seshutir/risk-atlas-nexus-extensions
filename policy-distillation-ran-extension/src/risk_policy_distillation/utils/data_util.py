import os
import random

import numpy as np
import torch
from datasets import load_dataset


def preprocess_dataset(dataset, prompt, response, label):
    dataset = dataset.rename(columns={prompt: 'prompt', response:'response', label: 'label'})
    if label == '':
        dataset['label'] = [1] * len(dataset)

    return dataset

def load_ds(dataconfig):
    name = dataconfig['general']['location']
    dataset = load_dataset(name)

    prompt = dataconfig['data']['prompt_col']
    response = dataconfig['data']['response']
    label = dataconfig['label']

    train_split = dataconfig['train_split']
    test_split = dataconfig['test_split']

    train_split = None if train_split == '' else train_split
    test_split = None if test_split == '' else test_split

    sample_ratio = dataconfig['sample_ratio']

    splits = generate_split(dataset, train_split, test_split, sample_ratio)

    for i in range(len(splits)):
        splits[i] = preprocess_dataset(splits[i], prompt, response, label)

    return splits


def generate_split(dataset, train_split, test_split, sample_ratio):
    if train_split is not None and test_split is not None:
        train = dataset.data[train_split].table.to_pandas()
        train = train.sample(int(sample_ratio*len(train)))

        train, val = split(train, (0.8, 0.2))
        test = dataset.data[test_split].table.to_pandas()
        test = test.sample(int(sample_ratio*len(test)))

        return [train, val, test]

    if train_split is None:
        test = dataset.data[test_split].table.to_pandas()
        test = test.sample(int(sample_ratio*len(test)))

        train, val, test = split(test, (0.6, 0.2, 0.2))

        return [train, val, test]

    if test_split is None:
        train = dataset.data[train_split].table.to_pandas()
        train = train.sample(int(sample_ratio*len(train)))

        train, val, test = split(train, (0.6, 0.2, 0.2))

        return [train, val, test]


def split(dataset, ratios):
    dataset = dataset.sample(frac=1, random_state=42)

    start = 0
    datasets = []
    for r in ratios:# Shuffle the DataFrame
        size = int(len(dataset) * r)
        d = dataset.iloc[start:(start+size)]
        start = start + size

        datasets.append(d)

    return datasets


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True