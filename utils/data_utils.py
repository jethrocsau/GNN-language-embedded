import os
from src.trainer_large import ModelTrainer, build_model
import numpy as np
import pandas as pd
import src.trainer_large as trainer_large
import torch
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from src.utils.utils import set_random_seed
import warnings
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dgl.dataloading import GraphDataLoader
import tqdm

# Global variables
dataset_names = ['ogbn-mag','ogbn-arxiv','ogbn-products']
utils_path = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(utils_path)
model_dir = os.path.join(cwd, 'model')
model_path = os.path.join(model_dir, 'GraphAlign_graphmae.pt')

# Model parameters
MODEL_NAME={"e5":"intfloat/e5-small-v2"}
MODEL_DIMS={"e5":384}

# get torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ogb_dataset(name):
    """
    Load  OGB dataset.
    Input: dataset name
    Output: graph, labels, train_idx, valid_idx, test_idx
    """

    data_dir = os.path.join(cwd, 'data')
    print(f"Dataset directory: {data_dir}")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if name not in dataset_names:
        raise ValueError(f"Dataset {name} not supported. Supported datasets are: {dataset_names}")
    else:
        dataset = DglNodePropPredDataset(name, root=data_dir)
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        return dataset, train_idx, valid_idx, test_idx


'''
Adapted from GraphAlign Github repository
'''

class GraphAlign_e5(ModelTrainer):
    def __init__(self, args):
        super().__init__(args)
        self._args = args
        self.pretrain_seed = 42
        self.dataset, self.train_idx, self.valid_idx, self.test_idx = load_ogb_dataset(args.dataset)
        self.graph, self.label = self.dataset[0]
        self.feat_type = getattr(args, 'feat_type', None)
        self.prepare_dataset()

    def load_model(self):
            set_random_seed(self.pretrain_seed)
            if self.feat_type == 'e5_float16':
                self._args.num_features = 384

            #build model
            self.model = build_model(self._args)
            self.model.to(self._args.device)

            #load model data
            print(f"Loading model from {self._args.load_model_path}")
            self.model.load_state_dict(torch.load(self._args.load_model_path))


    def infer_embeddings(self):
        args = self._args
        data_queue = queue.Queue(maxsize=15)
        self._eval_dataloader = GraphDataLoader(
            self.graph,
            self.train_idx,
            batch_size=args.batch_size_f,
            shuffle=False,
            drop_last=False
        )

        #num_info, label_info, self._eval_dataloader = load_dataloader("eval", args.dataset, args)
        # run inference on embedding
        with torch.no_grad():
            data_thread = threading.Thread(target=data_loading_thread, args=(data_queue,self._eval_dataloader,))
            data_thread.start()
            epoch_iter = tqdm(range(len(self._eval_dataloader)))
            self.model.to(self._device)
            self.model.eval()
            embeddings = []
            for idx in epoch_iter:
                batch = data_queue.get()
                batch_g, targets, _, node_idx = batch
                batch_g = batch_g.to(self._device)
                x = batch_g.ndata.pop("feat").to(self._device)
                targets = targets.to(self._device)
                batch_emb = self.model.embed(batch_g, x)[targets]
                embeddings.append(batch_emb.cpu())

        return torch.cat(embeddings, dim=0)

    def prepare_dataset(self):
        split_idx = self.dataset.get_idx_split()
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

def data_loading_thread(data_queue, dataloader):
    for batch in dataloader:
        data_queue.put(batch)



