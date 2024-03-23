import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../neuro')
sys.path.insert(0, '../../pyged/lib')
import time
import random
import pickle
import numpy as np
import torch
import torch.optim
import torch_geometric as tg
import torch_geometric.data
from tqdm.auto import tqdm
from neuro import config, datasets, metrics, models, train, utils
import pyged

dataset = 'Mutagenicity'
# innerfile = './aids_inner.pt
innerfile = ''
innerfile = f'./GED_{dataset}/test.pt'

outerfile = ''
# outerfile = f'./GED_{dataset}/outer_test.pt'



# Also need to change the method name from f2 to "ged_f2" if we want to use the GED method
config.method_name = ['ged_f2']
config.method_args = ['--threads 12 --time-limit 120']
config.n_workers = 12

# similarly, set the appropriate dist type('sed') or 'ged' to call the appropriate method to generate the data
dist_type = 'ged'

root_path = f'./GED_{dataset}/tmp'

if(dataset == 'AIDS'):
    graphs = utils.remove_extra_attrs(tg.datasets.TUDataset(root=root_path, name='AIDS700nef'))
    graphs = [g for g in graphs if utils.is_connected(g)]

elif(dataset == 'PubMed'):
    graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.CitationFull(root=root_path, name='PubMed')))

elif(dataset == 'Protein'):
    graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.TUDataset(root=root_path, name='PROTEINS_full')))
    graphs = [g for g in graphs if utils.is_connected(g)]

elif(dataset == 'CiteSeer'):
    graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.CitationFull(root=root_path, name='CiteSeer')))

elif(dataset == 'Cora_ML'):
    graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.CitationFull(root=root_path, name='Cora_ML')))

elif(dataset == 'AIDS700nef'):
    # DO NOT NEED TO REMOVE EXTRA ATTRS FOR AIDS700nef as we want to keep the node labels
    # graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.GEDDataset(root=root_path, name='AIDS700nef')))
    graphs = tg.datasets.GEDDataset(root=root_path, name='AIDS700nef')
    print("Dataset size: ", len(graphs))
    # graphs = utils.label_graphs(tg.datasets.GEDDataset(root=root_path, name='AIDS700nef'), num_classes=29)
    graphs = [g for g in graphs if utils.is_connected(g)]
    
elif(dataset == 'LINUX'):
    graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.GEDDataset(root=root_path, name='LINUX')))
    graphs = [g for g in graphs if utils.is_connected(g)]
    
elif dataset == 'IMDBMulti':
    graphs = utils.remove_extra_attrs(utils.label_graphs(tg.datasets.GEDDataset(root=root_path, name='IMDBMulti')))
    graphs = [g for g in graphs if utils.is_connected(g)]
    

elif dataset == "Mutagenicity":
    graphs = tg.datasets.TUDataset(root=root_path, name='Mutagenicity')
    # graphs = utils.remove_extra_attrs(tg.datasets.TUDataset(root=root_path, name='Mutagenicity'))
    print("Dataset size: ", len(graphs))
    graphs = [g for g in graphs if utils.is_connected(g)]

if innerfile:
    if dist_type == 'sed':
        train_set, train_meta = datasets.make_inner_dataset_plus(graphs, 500, 8, 0.5, node_lim_query=25, n_hops_target=None)
        torch.save((train_set, train_meta), innerfile)
    elif dist_type == 'ged':
        train_set = datasets.make_inner_dataset(graphs, 10000, 8, 0.5, node_lim_query=30, n_hops_target=None)
        torch.save(train_set, innerfile)

if outerfile:
    outer_test_set = datasets.make_outer_dataset(graphs, 2, 5, 0.6, node_lim_query=20, n_hops_target=None)
    torch.save(outer_test_set, outerfile)
