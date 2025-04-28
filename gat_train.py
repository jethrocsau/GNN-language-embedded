import gzip
import os
import pickle

import dgl
import numpy as np
import torch as th
from dgl.data.utils import load_graphs
from dgl.nn import GATConv

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
