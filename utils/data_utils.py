import os
import queue
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgl.dataloading import GraphDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.trainer_large import ModelTrainer, build_model
from src.utils.utils import set_random_seed

# Global variables
dataset_names = ['ogbn-mag','ogbn-arxiv','ogbn-products']
utils_path = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(utils_path)
model_dir = os.path.join(cwd, 'model')
data_dir = os.path.join(cwd, 'data')
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

    def get_nodeidx_mappings(self, return_val = True):
        node_idx = pd.read_csv(
            os.path.join(data_dir,'nodeidx2paperid.csv.gz'),
            sep=',', compression='gzip',
            header=0,
            names=['node_idx', 'paper_id']
        )
        node_idx['paper_id'] = node_idx['paper_id'].astype(np.int64)
        idx_title = pd.read_csv(
            os.path.join(data_dir,'titleabs.tsv.gz'),
            sep='\t',
            compression='gzip',
            header=0,
            names=['title', 'abstract'],
            dtype={'title': str, 'abstract': str}
        )
        self.node_titles = node_idx.merge(idx_title.reset_index(), left_on='paper_id', right_on='index', how='inner')
        if return_val:
            return self.node_titles


    def generate_e5_embeddings(self, return_val = True):
        # Load the E5 model and tokenizer
        input_texts = self.node_titles['title'].to_list()
        model_name = MODEL_NAME["e5"]
        model = SentenceTransformer(model_name, device = device)
        batch_size = 64
        embeddings = []
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Encoding Progress"):
            batch = input_texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        self.node_feat_e5 = torch.tensor(embeddings, dtype=torch.float16).to(device)
        if return_val:
            return self.node_feat_e5.cpu().numpy()

    def prepare_data(self):
        self.get_nodeidx_mappings(return_val=False)
        self.generate_e5_embeddings(return_val=False)
        self.graph.ndata['feat'] = self.node_feat_e5
        self.graph = self.graph.to(self._args.device)

    def infer_graphalign(self):
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


