import os
import pickle
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
stark_path = os.path.join(data_dir, 'stark-mag')

# mapping values
STARK_FILES = {
    "edge_index": "edge_index.pt",
    "edge_types": "edge_types.pt",
    "node_type_dict": "node_type_dict.pkl",
    "edge_type_dict": "edge_type_dict.pkl",
    "node_info": "node_info.pkl",
    "node_types": "node_types.pt"
}
MAPPING_FILES = {
    'ogbn-arxiv': os.path.join(data_dir, 'nodeidx2paperid.csv.gz'),
    'ogbn-mag': os.path.join(data_dir, 'paper_entidx2name.csv.gz')
}

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
        node_idx_file = MAPPING_FILES[self._args.dataset]
        node_idx = pd.read_csv(
            node_idx_file,
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
        self.node_titles['text'] = self.node_titles['title'] + ' ' + self.node_titles['abstract'].fillna('')
        input_texts = self.node_titles['text'].to_list()
        model_name = MODEL_NAME["e5"]
        model = SentenceTransformer(model_name, device = device)
        batch_size = 64
        embeddings = []
        for i in tqdm(range(0, len(input_texts), batch_size), desc="Encoding Progress"):
            batch = input_texts[i:i + batch_size]
            batch_embeddings = model.encode(batch, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
        self.node_feat_e5 = torch.tensor(embeddings, dtype=torch.float32).to(device)
        if return_val:
            return self.node_feat_e5.cpu().numpy()

    def prepare_data(self, return_val = True):
        self.load_model()
        self.get_nodeidx_mappings(return_val=False)
        self.generate_e5_embeddings(return_val=False)
        self.graph = self.graph.to(self._args.device)
        self.graph.ndata['e5_feat'] = self.node_feat_e5


    def infer_graphalign(self, return_val = True):
        self.model.eval()
        num_nodes = self.graph.num_nodes()
        batch_size = self._args.batch_size_f
        with torch.no_grad():
            #init features
            all_embeddings = []
            x = self.graph.ndata['e5_feat'].clone().to(self._device)

            # Process in batches
            for start_idx in tqdm(range(0, num_nodes, batch_size), desc="Processing batches"):
                end_idx = min(start_idx + batch_size, num_nodes)
                batch_nodes = torch.arange(start_idx, end_idx).to(self._device)
                batch_emb = self.model.embed(self.graph, x)[batch_nodes]
                all_embeddings.append(batch_emb.cpu())
            torch_embedding = torch.cat(all_embeddings, dim=0).to(self._device)

        # add embedding to graph
        self.graph = self.graph.to(self._args.device)
        self.graph.ndata['ga_embedding'] = torch_embedding
        if return_val:
            return torch_embedding.cpu().numpy()

    def prepare_dataset(self):
        split_idx = self.dataset.get_idx_split()
        self.train_idx, self.valid_idx, self.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

def data_loading_thread(data_queue, dataloader):
    for batch in dataloader:
        data_queue.put(batch)

def open_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

