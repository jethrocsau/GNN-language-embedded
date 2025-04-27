import os
import dgl
import numpy as np
import torch as th
from dgl.nn import GATConv
from dgl.data.utils import load_graphs

# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)

#get folders
processed_path = os.path.join(cwd, 'processed')
model_dir = os.path.join(cwd, 'model')

#load
arxiv_graph = os.path.join(processed_path, 'ogbn-arxiv_graph.bin')
arxiv_graphs, arxiv_label_dict = load_graphs(arxiv_graph) # glist will be [g1, g2]
arxiv_graph = arxiv_graphs[0]
arxiv_labels = arxiv_label_dict['glabel']

mag_graph = os.path.join(processed_path, 'ogbn-mag_graph.bin')
mag_graphs, mag_label_dict = load_graphs(mag_graph) # glist will be [g1, g2]
mag_graph = mag_graphs[0]
mag_labels = mag_label_dict['glabel']

# concat labels and ndata
training_feat = th.cat([arxiv_graph.ndata['feat'], mag_graph.ndata['feat']], dim=0)
labels = th.cat([arxiv_labels, mag_labels], dim=0)

# init net
input_dim = g.ndata['feat'].shape[1]
output_dim = labels.shape[1]
heads = 3

net = GATConv(
    in_feats=input_dim,
    out_feats=output_dim,
    num_heads=heads
)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print(
        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)
        )
    )

# save model after training


