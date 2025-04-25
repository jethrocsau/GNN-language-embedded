import os
from dgl.data.utils import load_graphs

# set params
env = 'terminal'
if env=='terminal':
    cwd = os.getcwd()
else:
    cwd = os.path.dirname(__file__)

#get folders
processed_path = os.path.join(cwd, 'processed')
processed_graph = os.path.join(processed_path, 'ogbn-arxiv_graph.bin')

#load
glist, label_dict = load_graphs(processed_graph) # glist will be [g1, g2]
g = glist[0]
labels = label_dict['glabel']

