import numpy as np
import pandas as pd
import os
from ogb.nodeproppred import DglNodePropPredDataset

dataset_names = ['ogbn-mag','ogbn-arxiv']
cwd = os.dir(os.path.dir(__file__))

def load_ogb_dataset(name, root_dir):
    """
    Load  OGB dataset.
    """

    return 0 