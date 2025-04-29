import gzip
import os
import pickle

import dgl
import numpy as np
import torch as th
from dgl.data.utils import load_graphs
from dgl.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import load_graphs

from utils.model import GAT, MLP, evaluate, train_model

#init dir
# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)

# device torch
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
th.manual_seed(42)

# set foldrs
data_dir = os.path.join(cwd, 'data')
processed_dir = os.path.join(cwd, 'processed')
graph_path = os.path.join(processed_dir,'combined_graph.bin')

# load graph
graphs, _ = load_graphs(graph_path)
graph = graphs[0]
labels = graph.ndata['label']

#load features
e5_features = graph.ndata['e5_feat']
ga_features = graph.ndata['ga_embedding']
original_features = graph.ndata['feat']


#get indices
train_idx = np.where(graph.ndata['train_mask'] == 1)[0]
val_idx = np.where(graph.ndata['valid_mask'] == 1)[0]
test_idx = np.where(graph.ndata['test_mask'] == 1)[0]

# init net
graph = dgl.add_self_loop(graph)
output_dim = labels.shape[1]

#adjust architecture by embedding type
# node2vec = 128
# e5_features = 384
# ga_features = 1024
gat_heads = 8
original_hidden_dim = 300
e5_hidden_dim = 100
ga_hidden_dim = 30
batch_size = 1024

# create model
gat_e5 = GAT(e5_features.shape[1],e5_hidden_dim , output_dim, gat_heads)
gat_ga = GAT(ga_features.shape[1],ga_hidden_dim , output_dim, gat_heads)
gat_original = GAT(original_features.shape[1],original_hidden_dim , output_dim, gat_heads)

# Move models and data `to device
gat_e5 = gat_e5.to(device)
gat_ga = gat_ga.to(device)
gat_original = gat_original.to(device)
graph = graph.to(device)
e5_features = e5_features.to(device)
ga_features = ga_features.to(device)
original_features = original_features.to(device)
labels = labels.to(device)


# train modelx
gat_e5_state, gat_e5_best_val = train_model(
   gat_e5,
   graph,
   e5_features,
   labels,
   train_idx,
   val_idx,
   epochs=200,
   lr=0.005
)
gat_e5.load_state_dict(gat_e5_state)

gat_ga_state, gat_ga_best_val = train_model(
   gat_ga,
   graph,
   ga_features,
   labels,
   train_idx,
   val_idx,
   epochs=200,
   lr=0.005
)
gat_ga.load_state_dict(gat_ga_state)

gat_original_state, gat_original_best_val = train_model(
   gat_original,
   graph,
   original_features,
   labels,
   train_idx,
   val_idx,
   epochs=200,
   lr=0.005
)
gat_original.load_state_dict(gat_original_state)

# evaluate
gat_e5_accuracy, gat_e5_f1 = evaluate(
   gat_e5,
   graph,
   e5_features,
   labels,
   graph.ndata['test_mask']
)
gat_ga_accuracy, gat_ga_f1 = evaluate(
   gat_ga,
   graph,
   ga_features,
   labels,
   graph.ndata['test_mask']
)
gat_original_accuracy, gat_original_f1 = evaluate(
   gat_original,
   graph,
   original_features,
   labels,
   graph.ndata['test_mask']
)

# print results
print(f"GAT E5 Test Accuracy: {gat_e5_accuracy:.4f}, F1: {gat_e5_f1:.4f}")
print(f"GAT GA Test Accuracy: {gat_ga_accuracy:.4f}, F1: {gat_ga_f1:.4f}")
print(f"GAT Original Test Accuracy: {gat_original_accuracy:.4f}, F1: {gat_original_f1:.4f}")

# save results
results = {
   'gat_e5_accuracy': gat_e5_accuracy,
   'gat_e5_f1': gat_e5_f1,
   'gat_ga_accuracy': gat_ga_accuracy,
   'gat_ga_f1': gat_ga_f1,
   'gat_original_accuracy': gat_original_accuracy,
   'gat_original_f1': gat_original_f1,
}
with open(os.path.join(processed_dir, 'gat_results.pkl'), 'wb') as f:
   pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

# save model
th.save(gat_e5.state_dict(), os.path.join(processed_dir, 'gat_e5_model.pth'))
th.save(gat_ga.state_dict(), os.path.join(processed_dir, 'gat_ga_model.pth'))
th.save(gat_original.state_dict(), os.path.join(processed_dir, 'gat_original_model.pth'))

