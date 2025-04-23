import os

import numpy as np
import pandas as pd
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

dataset_names = ['ogbn-mag','ogbn-arxiv']
cwd = os.dir(os.path.dir(__file__))


def load_ogb_dataset(name):
    """
    Load  OGB dataset.
    Input: dataset name
    Output: graph, labels, train_idx, valid_idx, test_idx
    """

    data_dir = os.path.join(cwd, 'data')
    if not os.path.exist(data_dir):
        os.makedirs(data_dir)

    if name not in dataset_names:
        raise ValueError(f"Dataset {name} not supported. Supported datasets are: {dataset_names}")
    else:
        dataset = DglNodePropPredDataset(name, root=os.path.join(cwd, 'data'))
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]
        return graph, label, train_idx, valid_idx, test_idx





