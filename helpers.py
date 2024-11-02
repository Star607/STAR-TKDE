import torch
from torch.autograd import Variable
from math import ceil
import numpy as np
import random
import logging
from datetime import datetime
import sys
import os
from pathlib import Path 

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_logger(log_dir='data/', log_prefix='STAR'):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s',
        "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    ts = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    fh = logging.FileHandler(f'{log_dir}/{log_prefix}-{ts}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        print("bool value expected.")

def write_result(JSDs, dataset, params, postfix="STAR-TKDE", res_path="results"):
    Path(res_path).mkdir(parents=True, exist_ok=True)
    res_path = "{}/{}-{}.csv".format(res_path, dataset, postfix)
    headers = ["method", "dataset", 'dis', 'rad', 'dur', 'dloc', 'grk', 'irk', "params"]

    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            postfix, dataset, JSDs[0], JSDs[1], JSDs[2], JSDs[3], JSDs[4], JSDs[5])
        logging.info(result_str)
        params_str = ",".join(["{}={}".format(k, v)
                               for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        f.write(row)

def read_data_from_file(fp):
    path = []
    with open(fp, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pois = line.split(' ')
            path.append([int(poi) for poi in pois])
    return path

def get_gps(gps_file):
    gps = np.load(gps_file)
    X, Y= gps[:,0], gps[:,1]
    return X, Y

def add_eos_and_pad_seq(seqs, EOS = None, mode = 'no-eos'):
    max_seq = 24
    valid_len = [len(seq) for seq in seqs]
    for i, seq in enumerate(seqs):
        if valid_len[i] < max_seq:
            if mode == 'add-eos':
                seq.append(EOS)
                valid_len[i] += 1
                if valid_len[i] < max_seq:
                    seq.extend([0] * (max_seq - valid_len[i]))
            else:
                seq.extend([0] * (max_seq - valid_len[i]))
        assert len(seq) == max_seq
    return seqs, valid_len

def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def prepare_generator_batch(samples, device):
    x_seq, y_seq = torch.zeros_like(samples), torch.zeros_like(samples)
    b, t = samples.shape[0], samples.shape[1]-1
    x_seq, y_seq = torch.zeros((b, t)).to(samples), torch.zeros((b, t)).to(samples)
    x_seq[:, :] = samples[:, :-1]
    y_seq[:, :] = samples[:, 1:]
    return x_seq, y_seq

def prepare_discriminator_data(pos_samples, pos_len, neg_samples, neg_len, device):
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0
    lens = torch.cat((pos_len, neg_len)).type(torch.IntTensor)
    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]
    lens = lens[perm]

    inp = Variable(inp).to(device)
    target = Variable(target).to(device)
    lens = Variable(lens).to(device)
    return inp, target, lens

def batchwise_sample(gen, num_samples, batch_size):
    samples = []
    samples_len = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        s, s_len = gen.sample(batch_size)
        samples.append(s)
        samples_len.append(s_len)
    return torch.cat(samples, 0)[:num_samples], torch.cat(samples_len, 0)[:num_samples]

def sample(real_data_samples, data_lens, num_samples):
        sample_idx = torch.randint(len(real_data_samples), size = (num_samples,))
        return real_data_samples[sample_idx], data_lens[sample_idx]
    
def batchwise_nll(gen, real_data_samples, num_samples, data_lens, batch_size, device):
    gen_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(real_data_samples[i:i+batch_size], device)
        lens = data_lens[i:i+batch_size] - 1
        gen_nll += gen.batchNLLLoss(inp, target, lens.to(device)).data.item()
    return gen_nll/(num_samples/batch_size)