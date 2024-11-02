import argparse
import os
import sys
import time
from math import ceil
from pathlib import Path
import socket
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import discriminator
import generator
import helpers

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def train_generator_MLE(args, gen, gen_opt, real_data_samples, data_lens, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    batch_num = int(np.ceil(len(real_data_samples)/args.batch_size))
    for epoch in range(epochs):
        total_loss = 0
        start_i = 0
        for i in range(batch_num): 
            end_i = min(start_i + args.batch_size, len(real_data_samples))
            inp, target = helpers.prepare_generator_batch(real_data_samples[start_i:end_i], device=args.device)
            gen_opt.zero_grad()
            lens = data_lens[start_i:end_i] - 1
            loss = gen.batchNLLLoss(inp, target, lens.to(args.device)) 
            loss.backward()
            gen_opt.step()
            total_loss += loss.data.item() 
            start_i += args.batch_size

        total_loss = total_loss / float(batch_num)
        train_gen_loss = helpers.batchwise_nll(gen, real_data_samples, len(real_data_samples), data_lens, args.batch_size, args.device)
        logger.info('Epoch %d average_train_NLL = %.4f, gen_sample_NLL = %.4f' % (epoch+1, total_loss, train_gen_loss))

def train_generator_PG(args, train_data, train_len, gen, gen_opt, dis, num_batches, epoch):
    for batch in range(num_batches):
        s, s_len = gen.sample(args.batch_size*2)     
        rewards = dis.batchClassify(s, s_len) 
        inp, target = helpers.prepare_generator_batch(s, device=args.device)
        lens = s_len - 1
        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards, lens) 
        pg_loss.backward()
        gen_opt.step()

    train_loss = helpers.batchwise_nll(gen, train_data, len(train_data), train_len, args.batch_size, args.device)
    logger.info('train_sample_NLL = %.4f' % train_loss)

def train_discriminator(args, discriminator, dis_opt, real_data_samples, data_lens, generator, d_steps, epochs):
    pos_val, pos_val_len = helpers.sample(real_data_samples, data_lens, 100) 
    neg_val, neg_val_len = generator.sample(100)
    val_inp, val_target, val_len = helpers.prepare_discriminator_data(pos_val, pos_val_len, neg_val, neg_val_len, device=args.device)
    for d_step in range(d_steps):
        s, s_len = helpers.batchwise_sample(generator, len(real_data_samples), args.batch_size) 
        dis_inp, dis_target, dis_len = helpers.prepare_discriminator_data(real_data_samples, data_lens, s, s_len, device=args.device)  
        for epoch in range(epochs):
            logger.info('d-step %d epoch %d : ' % (d_step + 1, epoch + 1))
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0
            for i in range(0, 2 * len(real_data_samples), args.batch_size): 
                inp, target, inp_len = dis_inp[i:i + args.batch_size], dis_target[i:i + args.batch_size], dis_len[i:i + args.batch_size]
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp, inp_len)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)/len(inp)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>=0.5)==(target>=0.5)).data.item()

            total_loss /= ceil(2 * len(real_data_samples) / float(args.batch_size))
            total_acc /= float(2 * len(real_data_samples))

            val_pred = discriminator.batchClassify(val_inp, val_len)
            logger.info('average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>=0.5)==(val_target>=0.5)).data.item()/200.))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  default=0, type=int, choices=[0,1,2,3,4])   
    parser.add_argument('--cuda',  default="0", type=str)   
    parser.add_argument('--device',  default="cpu", type=str)   
    parser.add_argument('--data', type=str, default='NYC', choices=['geolife', 'gowalla', 'brightkite', 'NYC', 'TKY', 'Moscow', 'Singapore'])
    parser.add_argument('--min_seq_len', default='8', type=int) 
    parser.add_argument('--max_seq_len', default='24', type=int) 
    parser.add_argument('--method', default='STAR-TKDE', type=str)  
    parser.add_argument('--train_ratio', default='0.7', type=float) 
    parser.add_argument('--val_ratio', default='0.1', type=float) 
    parser.add_argument('--test_ratio', default='0.2', type=float)
    parser.add_argument('--batch_size', default='32', type=int) 
    parser.add_argument('--mle_train_epochs', default='100', type=int)  
    parser.add_argument('--adv_train_epochs', default='50', type=int)  
    parser.add_argument('--gen_embedding_dim', default='32', type=int)
    parser.add_argument('--gen_hidden_dim', default='32', type=int)
    parser.add_argument('--dis_embedding_dim', default='64', type=int)
    parser.add_argument('--dis_hidden_dim', default='64', type=int)
    parser.add_argument('--lr', default=1e-2, type=float)

    parser.add_argument('--node_fea_dim', default='32', type=int) 
    parser.add_argument('--g_hid_dim', default='32', type=int, choices=[16, 32, 48, 64])
    parser.add_argument('--g_out_dim', default='32', type=int, choices=[16, 32, 48, 64])
    parser.add_argument('--graph_channels', default='3', type=int)
    parser.add_argument('--head_num', default='2', type=int, choices=[1, 2, 4, 8]) 
    parser.add_argument('--layer_num', default='2', type=int, choices=[1, 2, 3])

    parser.add_argument('--edge_type', default='adjacent', type=str) 
    parser.add_argument('--emb_type', default='default', type=str) 
    parser.add_argument('--gid', default='012', type=str)   
    parser.add_argument('--p_stay', default='learnable', type=str) 
    parser.add_argument('--decay_rate', default=1, type=int) 
    parser.add_argument('--gmodel', default='GAT', type=str, choices=['GAT', 'GCN', 'SAGE']) 
    parser.add_argument('--gene_num', default=5000, type=int) 
    
    args = parser.parse_args()
    helpers.set_random_seed(args.seed)
    args.hostname = socket.gethostname()

    all_name = ['sdg', 'ttg', 'stg'] 
    gid2name = {'012': all_name[:]}

    graph_name = gid2name[args.gid]
    print('graph name: ', graph_name)
    args.graph_channels = len(graph_name)
    args.device = torch.device("cuda:" + args.cuda)
    args.g_out_dim = args.g_hid_dim
    args.gen_embedding_dim = args.g_hid_dim
    args.gen_hidden_dim = args.g_hid_dim

    args.cwd = os.path.dirname(os.getcwd())
    args.datapath = f'{args.cwd}/gen-val-data/tra{args.train_ratio}-val{args.val_ratio}-test{args.test_ratio}/min_len_{args.min_seq_len}/{args.data}'
    args.savepath = f'{args.cwd}/{args.method}/out'
    args.propath = f'saved_data/{args.data}' 
    train_str = f'seed{args.seed}-{args.edge_type}-{args.emb_type}-gid{args.gid}-p{args.p_stay}-dc{args.decay_rate}-h{args.head_num}-l{args.layer_num}-lr{args.lr}-bs{args.batch_size}-gnum{args.gene_num}'
    Path(args.savepath).mkdir(parents=True, exist_ok=True)
    path_train_model = f'{args.savepath}/{args.data}_{train_str}_ckpt.pth'
    path_train_model_last = f'{args.savepath}/{args.data}_{train_str}_ckpt_last.pth' 

    params_map = {'TKY':{'num_locs':2862, 'max_dist':0.4353543817996979},
                "NYC":{'num_locs':1341, 'max_dist':0.5631671075636575},
                "gowalla":{'num_locs':32543, 'max_dist':340.3848501438751},
                "brightkite":{'num_locs':30412, 'max_dist':359.6742958009268},
                "geolife":{'num_locs':8462, 'max_dist':355.1678724707548},
                "Singapore":{'num_locs':4940, 'max_dist':285.4068298339844},
                "Moscow":{'num_locs':5052, 'max_dist':258.0411376953125}}

    args.num_locs = params_map[args.data]['num_locs']
    args.max_dist = params_map[args.data]['max_dist']

    params = {'seed': args.seed, 'edge': args.edge_type, 'emb': args.emb_type, "gid": args.gid, "p_stay": args.p_stay, 'decay_rate': args.decay_rate, "head_num": args.head_num, "layer_num": args.layer_num, 'batch_size': args.batch_size, 'lr': args.lr, 'gene_num': args.gene_num}


    log_dir = f'./logs'
    log_prefix=f'{args.method}-main-{args.seed}-{args.data}-{args.hostname}-gpu{args.cuda}'
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
    logger.info(args)

    logger.info('Model init.')
    gen = generator.GraphGenerator(args.gen_embedding_dim, args.gen_hidden_dim, args.num_locs, args.min_seq_len, args.max_seq_len, device=args.device, len_list=np.load(f'{args.datapath}/train_len.npy'), starting_dist=np.load(f'{args.datapath}/start.npy'), node_fea_dim = args.node_fea_dim, propath=args.propath, edge_type = args.edge_type, node_emb_init_type=args.emb_type, graph_names=graph_name, graph_channels=args.graph_channels, p_stay=args.p_stay, gmodel=args.gmodel, g_hid_dim=args.g_hid_dim, g_out_dim=args.g_out_dim, head_num=args.head_num, layer_num=args.layer_num, decay_rate=args.decay_rate)
    
    dis = discriminator.Discriminator(args.dis_embedding_dim, args.dis_hidden_dim, args.num_locs, args.min_seq_len, args.max_seq_len, device=args.device)
    gen = gen.to(args.device)
    dis = dis.to(args.device)

    logger.info('Loading data.')
    train_data = helpers.read_data_from_file(f'{args.datapath}/train.txt') 
    train_data, train_len = helpers.add_eos_and_pad_seq(train_data)
    train_data, train_len = torch.tensor(train_data).to(args.device), torch.tensor(train_len).to(args.device)

    val_data = helpers.read_data_from_file(f'{args.datapath}/val.txt') 
    valid_data, valid_len = helpers.add_eos_and_pad_seq(val_data)
    valid_data, valid_len = torch.tensor(valid_data).to(args.device), torch.tensor(valid_len).to(args.device)
    
    start = time.time()
    # Pretrain
    logger.info('Starting Generator MLE Training')
    gen_optimizer = optim.Adam(gen.parameters(), lr=args.lr)
    train_generator_MLE(args, gen, gen_optimizer, train_data, train_len, args.mle_train_epochs) 

    logger.info('Starting Discriminator Training.')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(args, dis, dis_optimizer, train_data, train_len, gen, 50, 3) 

    # Adversarial training
    logger.info(f'Starting Adversarial Training.')
    loss = helpers.batchwise_nll(gen, train_data, len(train_data), train_len, args.batch_size, args.device) 
    logger.info('Initial train_NLL : %.4f' % loss) 
 
    best_val_loss = 1e9
    for epoch in range(args.adv_train_epochs):
        logger.info('EPOCH %d------------------------' % (epoch+1))
        train_generator_PG(args, train_data, train_len, gen, gen_optimizer, dis, 1, epoch)
        train_discriminator(args, dis, dis_optimizer, train_data, train_len, gen, 5, 3)
    
        valid_loss = helpers.batchwise_nll(gen, valid_data, len(valid_data), valid_len, args.batch_size, args.device)
        logger.info('EPOCH %2d: valid_NLL = %.4f' % (epoch, valid_loss))
        if best_val_loss > valid_loss:
            best_val_loss = valid_loss
            if epoch > 0:
                ckpt = {
                    'gen': gen.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis': dis.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),                   
                    'model_args': params,
                    'epoch': epoch,
                    'best_val_acc_top5': best_val_loss
                }
                logger.info('Save checkpoint of model: %s', path_train_model)
                torch.save(ckpt, path_train_model)

        if epoch == args.adv_train_epochs - 1:
            pred_data = gen.sample(args.gene_num, mode = 'no-padding')
            with open(f'{args.savepath}/{args.data}_{train_str}_gene{epoch}.txt', 'w') as f:
                for path in pred_data:
                    f.write(' '.join([str(poi) for poi in path]))
                    f.write('\n')
            ckpt_last = {
                    'gen': gen.state_dict(),
                    'gen_optimizer': gen_optimizer.state_dict(),
                    'dis': dis.state_dict(),
                    'dis_optimizer': dis_optimizer.state_dict(),                   
                    'model_args': params,
                    'epoch': epoch,
                    'best_val_acc_top5': best_val_loss
            }
            logger.info('Save checkpoint of models: %s', path_train_model_last)
            torch.save(ckpt_last, path_train_model_last)

