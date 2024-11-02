import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path 
from tqdm import tqdm
import networkx as nx
import logging
import scipy.sparse.linalg
import pandas as pd
import os
import helpers
import socket
from scipy.stats import wasserstein_distance
import numba as nb
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz

parser = argparse.ArgumentParser()
parser.add_argument('--cuda',  default="0", type=str)   
parser.add_argument('--data', type=str, default='NYC', choices=['gowalla', 'brightkite', 'NYC', 'TKY', 'Moscow', 'Singapore'])
parser.add_argument('--min_seq_len', default='8', type=int) 
parser.add_argument('--max_seq_len', default='24', type=int) 
parser.add_argument('--method', default='STAR-TKDE', type=str)  
parser.add_argument('--train_ratio', default='0.7', type=float) 
parser.add_argument('--val_ratio', default='0.1', type=float) 
parser.add_argument('--test_ratio', default='0.2', type=float)
parser.add_argument('--eigen_dim', default='32', type=int)
parser.add_argument("--ratio", type=float, default=0.005, help="sparsity of spatial graph")
parser.add_argument('--eigen', action="store_true")   
args = parser.parse_args()

params_map = {'TKY':{'num_locs':2862, 'max_dist':0.4353543817996979},
                "NYC":{'num_locs':1341, 'max_dist':0.5631671075636575},
                "gowalla":{'num_locs':32543, 'max_dist':340.3848501438751},
                "brightkite":{'num_locs':30412, 'max_dist':359.6742958009268},
                "geolife":{'num_locs':8462, 'max_dist':355.1678724707548},
                "Singapore":{'num_locs':4940, 'max_dist':285.4068298339844},
                "Moscow":{'num_locs':5052, 'max_dist':258.0411376953125}}
args.num_locs = params_map[args.data]['num_locs']
args.device = torch.device("cuda:" + args.cuda)
args.hostname = socket.gethostname()
args.cwd = os.path.dirname(os.getcwd())
args.datapath = f'{args.cwd}/gen-val-data/tra{args.train_ratio}-val{args.val_ratio}-test{args.test_ratio}/min_len_{args.min_seq_len}/{args.data}'
args.propath = f'saved_data/{args.data}' 
Path(args.propath).mkdir(parents=True, exist_ok=True)

log_dir = f'./logs'
log_prefix=f'{args.method}-nfeat-{args.data}-{args.hostname}-gpu{args.cuda}'
Path(log_dir).mkdir(parents=True, exist_ok=True)
logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
logger.info(args)

train_data = helpers.read_data_from_file(f'{args.datapath}/train.txt') 
train_t = helpers.read_data_from_file(f'{args.datapath}/train_t.txt')
gps = np.load(f'{args.datapath}/gps.npy')
top = int(args.ratio * args.num_locs)

# temporal transition graph 
if os.path.exists(f'{args.propath}/ttg.npz'):
    logger.info('load temporal transition graph...')
    ttg = load_npz(f'{args.propath}/ttg.npz')
else:
    ttg_all = np.zeros([args.num_locs, args.num_locs]) 
    for seq in train_data:
        for j in range(len(seq)-1):
            ttg_all[seq[j], seq[j+1]] += 1
    for i in range(args.num_locs): # self loop
        ttg_all[i, i] = max(1, ttg_all[i, i])
    save_npz(f'{args.propath}/ttg_all.npz', csr_matrix(ttg_all))
    logging.info(f"save ttg_all.npz in {args.propath}")

    _ttg = np.zeros([args.num_locs, args.num_locs]) 
    _ttg_all = ttg_all + 1e-10
    ttg_all_norm = _ttg_all / _ttg_all.sum(axis=1)[:, None]  
    for i in range(len(ttg_all_norm)):
        top_id = ttg_all_norm[i,:].argsort()[-top:]
        for id in top_id: 
            _ttg[i, id] = ttg_all_norm[i, id] 
    ttg = csr_matrix(_ttg)
    save_npz(f'{args.propath}/ttg.npz', ttg)
    logging.info(f"save ttg.npz in {args.propath}")

logger.info(f"Sparsity of ttg: {(1.0*len(ttg.data)/(args.num_locs * args.num_locs)):.5f}")

# spatial distance graph
if os.path.exists(f'{args.propath}/sdg.npz'):
    logger.info('load spatial distance graph...')
    sdg = load_npz(f'{args.propath}/sdg.npz')
else:
    latitude, longitude = gps[:, 0], gps[:, 1]
    lgti = np.radians(longitude)[:, np.newaxis] 
    lati = np.radians(latitude)[:, np.newaxis]
    
    @nb.njit(nogil=True, parallel=True)
    def np_distance(lgti, lati):
        distances = np.zeros((len(lgti), len(lgti))).astype(np.float32)
        for i in nb.prange(len(lgti)):
            gi, ai = lgti[i], lati[i]
            dlon = gi - lgti  
            dlat = ai - lati
            dist = dlon**2 + dlat**2
            distances[i] = np.sqrt(np.reshape(dist, (len(dist))))
        return distances
    sdg_all = np_distance(lgti, lati)
    np.save(f'{args.propath}/sdg_all.npy', sdg_all)

    _sdg = np.zeros([args.num_locs, args.num_locs]) 
    _sdg_all = sdg_all + 1e-10
    sdg_all_norm = _sdg_all / _sdg_all.sum(axis=1, keepdims=True)  
    sdg_all_sim = 1 / sdg_all_norm  
    sdg_all_sim_norm =  sdg_all_sim / sdg_all_sim.sum(axis=1, keepdims=True)
    for i in range(len(sdg_all_sim_norm)):
        top_id = sdg_all_sim_norm[i,:].argsort()[-top:]
        assert i in top_id
        for id in top_id: 
            _sdg[i, id] = sdg_all_sim_norm[i, id] 
    sdg = csr_matrix(_sdg)
    save_npz(f'{args.propath}/sdg.npz', sdg)
    logging.info(f"save sdg.npz in {args.propath}")
logger.info(f"Sparsity of sdg: {(1.0*len(sdg.data)/(args.num_locs * args.num_locs)):.5f}")

# spatiotemporal graph
if os.path.exists(f'{args.propath}/stg.npz'):
    logger.info('load spatiotemporal graph...')
    stg = load_npz(f'{args.propath}/stg.npz')
else:
    poi_dis = np.zeros((args.num_locs, args.max_seq_len))
    for i in tqdm(range(len(train_data))): 
        for poi, t in zip(train_data[i], train_t[i]):
            poi_dis[poi][t] += 1
    poi_dis = (poi_dis + 1) / (poi_dis.sum(axis=1, keepdims=True) + poi_dis.shape[1]) # add-one smooth
    _poi_sim = np.zeros((args.num_locs, args.num_locs))
    bins = np.arange(args.max_seq_len)
    for i in tqdm(range(args.num_locs)):
        for j in range(i+1, args.num_locs):
            _poi_sim[i, j] = 1 - wasserstein_distance(bins, bins, poi_dis[i], poi_dis[j])
    poi_sim_sym = _poi_sim + _poi_sim.T
    stg_all = poi_sim_sym + np.identity(args.num_locs)
    np.save(f'{args.propath}/stg_all.npy', stg_all)

    _stg = np.zeros((args.num_locs, args.num_locs)) 
    for i in range(stg_all.shape[0]):
        top_id = stg_all[i,:].argsort()[-top:]
        assert len(top_id) > 0
        for id in top_id:
            _stg[i, id] = stg_all[i, id]
    stg = csr_matrix(_stg)
    save_npz(f'{args.propath}/stg.npz', stg)
    logging.info(f"save stg.npz in {args.propath}")
logger.info(f"Sparsity of stg: {(1.0*len(stg.data)/(args.num_locs * args.num_locs)):.5f}")

adj_l = [ttg, sdg, stg]
name_l = ['ttg', 'sdg', 'stg']
for name, adj in zip(name_l, adj_l):
    e_src, e_dst = adj.nonzero()
    weights = adj.data
    e_and_w = np.hstack((e_src[:, None], e_dst[:, None], weights[:, None]))

    if args.eigen:
        edge_tuples = [(int(u), int(v), {'weight': w}) for u, v, w in e_and_w]
        nx_g = nx.Graph(edge_tuples) 
        logging.info('Eigen vector decomposition')
        lap = nx.laplacian_matrix(nx_g).asfptype()
        w, v = scipy.sparse.linalg.eigs(lap, k=args.eigen_dim)
        logger.info(f'node num: {args.num_locs}, edge num: {len(edge_tuples)}, node feature shape: {v.shape}')
        eigen_vectors = np.real(v)
    else:
        eigen_vectors = np.zeros((args.num_locs, args.eigen_dim))
    logger.info('eigen_vectors done.')

    central_dict = dict()
    nodes = pd.DataFrame(data = range(args.num_locs), columns = ['node_id'])
    reorder_eigen_vectors = eigen_vectors[nodes['node_id'].tolist()]
    for i in range(args.eigen_dim):
        col_name = 'eigen_vec{:03d}'.format(i)
        central_dict[col_name] = reorder_eigen_vectors[:, i]

    for col in central_dict:
        nodes[col] = central_dict[col]
    nodes.to_csv(f'{args.propath}/fea_dim_{args.eigen_dim}_{name}_nodes.csv', index=None)

    edges = pd.DataFrame(data = e_and_w, columns = ['src', 'dst', 'weight'])
    edges.to_csv(f'{args.propath}/fea_dim_{args.eigen_dim}_{name}_edges.csv', index=None)
    logging.info('save nodes and edges.')
    logger.info('save files done.')
