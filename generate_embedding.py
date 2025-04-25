import os
from argparse import Namespace

import torch
from dgl.dataloading import GraphDataLoader

from utils.data_utils import GraphAlign_e5, load_ogb_dataset, open_pickle

# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)
model_path = os.path.join(cwd, 'model', 'GraphAlign_graphmae.pt')
config_path = os.path.join(cwd, 'src', 'config', 'GraphMAE_configs.yml')
data_dir = os.path.join(cwd, 'data')

# cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# datset names
dataset_names = ['ogbn-mag','ogbn-arxiv','ogbn-products']
idx_dataset = 2

# set arguments
args = Namespace(
    dataset=dataset_names[idx_dataset],
    model='graphmae',
    use_cfg=True,
    device=0,
    load_model=True,
    load_model_path=model_path,
    data_dir=data_dir,
    feat_type='e5_float16',
    use_cfg_path=config_path,
    prob_num_workers=4,
    moe=True,
    moe_use_linear=True,
    top_k=1,
    num_expert=4,
    hiddenhidden_size_times=1,
    moe_layer=[0],
    decoder_no_moe=True,
    num_hidden=1024,
    num_layers=4,
    num_proj_hidden=1024,
    num_heads=4,
    encoder='gat',
    activation='prelu',
    norm='layernorm',
    num_out_heads=1,
    in_drop=0.2,
    attn_drop=0.2,
    feat_drop=0.2,
    negative_slope=0.2,
    residual=True,
    mask_rate=0.5,
    replace_rate=0.0,
    alpha_l=2,
    weight_decay_f=1e-4,
    lr_f=0.001,
    max_epoch_f=1000,
    batch_size_f=256,
    scheduler=True,
    loss_fn='sce',
    optimizer='adamw',
    linear_prob=True,
    dropout=0.2,
    decoder='gat',
    drop_edge_rate=0.0,
    concat_hidden=False,
    deepspeed=False,
    lr=0.001,
    weight_decay=0.0001,
    num_workers=4,
    num_epochs=1000,
    no_verbose=False,
    save_model=True
)

# create & load model
model = GraphAlign_e5(args)
model.load_model()

# get embeddings
node_titles = model.get_nodeidx_mappings()
node_feat_e5 = model.generate_e5_embeddings()


# save embedding
#if not os.path.exists(os.path.join(cwd, 'embedding')):
#    os.makedirs(os.path.join(cwd, 'embedding'))
#    torch.save(embedding, os.path.join(cwd, 'embedding', 'embedding.pt'))
