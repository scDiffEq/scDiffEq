# training and evaluation for interpolation and fate prediction tasks
# on weinreb et al. dataset

import torch
import torch.nn.functional as F 
from torch import nn, optim

import annoy
import tqdm

from geomloss import SamplesLoss

import numpy as np
import pandas as pd
import scipy.stats

from collections import OrderedDict, Counter
from joblib import Parallel, delayed 
from types import SimpleNamespace 
from time import strftime, localtime 

import argparse
import copy
import glob
import itertools
import json 
import os
import sys

# import train
import prescient.train as train

# FATE_DIR = "data/Klein2020_fate"
# FATE_TRAIN_PATH= os.path.join(FATE_DIR, "fate_train.pt")
# FATE_ANN = os.path.join(FATE_DIR, "50_20_10")
# FATE_TEST_PATH = os.path.join(FATE_DIR, "fate_test.pt")

IMPUTE_DATA_PATH = "../data/Weinreb2020_impute.pt"
WEIGHT_DIR = '../data/'

def init_config(args): 

    config = SimpleNamespace(

        seed = args.seed,
        timestamp = strftime("%a, %d %b %Y %H:%M:%S", localtime()), 

        # data parameters
        data_dir = args.data_dir, 
        data_path = args.data_path,
        weight_path = args.weight_path,
        weight = args.weight, 

        # model parameters 
        activation = args.activation,
        layers = args.layers, 
        k_dim = args.k_dim, 

        # pretraining parameters
        pretrain_burnin = 50, 
        pretrain_sd = 0.1, 
        pretrain_lr = 1e-9, 
        pretrain_epochs = args.pretrain_epochs, 

        # training parameters
        train_dt = args.train_dt, 
        train_sd = args.train_sd, 
        train_batch_size = args.train_batch, 
        ns = 2000,
        train_burnin = 100, 
        train_tau = args.train_tau, 
        train_epochs = args.train_epochs, 
        train_lr = args.train_lr, 
        train_clip = args.train_clip,
        save = args.save,

        # loss parameters
        sinkhorn_scaling = 0.7, 
        sinkhorn_blur = 0.1, 

        # file parameters
        out_dir = args.out_dir, 
        out_name = args.out_dir.split('/')[-1], 
        pretrain_pt = os.path.join(args.out_dir, 'pretrain.pt'), 
        train_pt = os.path.join(args.out_dir, 'train.{}.pt'), 
        train_log = os.path.join(args.out_dir, 'train.log'), 
        done_log = os.path.join(args.out_dir, 'done.log'),
        config_pt = os.path.join(args.out_dir, 'config.pt'),
    )

    config.train_t = []
    config.test_t = []
   
    if not os.path.exists(args.out_dir):
        print('Making directory at {}'.format(args.out_dir))
        os.makedirs(args.out_dir)
    else: 
        print('Directory exists at {}'.format(args.out_dir))

    return config

def load_data(config, base_dir = "."): 
   
    data_pt = torch.load(os.path.join(base_dir, config.data_path))
    x = data_pt['xp'] 
    y = data_pt['y']

    config.x_dim = x[0].shape[-1]
    config.t = y[-1] - y[0]
    
    y_start = y[config.start_t]
    y_ = [y_ for y_ in y if y_ > y_start]

    weight_pt = torch.load(os.path.join(base_dir, config.weight_path))
    x = [xx[m] for xx, m in zip(x, weight_pt['m'])]

    w_ = weight_pt['w'][config.start_t]
    w = {(y_start, yy): torch.from_numpy(np.exp((yy - y_start)*w_)) for yy in y_}

    return x, y, w

def train_fate(args): 

    a = copy.copy(args)

    # data

    a.data_path = os.path.join(a.data_dir, 'fate_train.pt')

    weight = os.path.basename(a.weight_path)
    weight = weight.split('.')[0].split('-')[-1]
    a.weight = weight

    # out directory

    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_tau}"
    ).format(**a.__dict__)

    a.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(a.seed))
    config = init_config(a)
    
    config.start_t = 0
    config.train_t = [1, 2]
    
    x, y, w = load_data(config)

    return x, y, w, config

def evaluate_fate(args, config): 

    # -- load data
 
    data_pt = torch.load(os.path.join(config.data_dir, 'fate_test.pt'))
    x = data_pt['x']
    y = data_pt['y']
    t = data_pt['t']

    ay_path = os.path.join(config.data_dir, '50_20_10')
    ay = annoy.AnnoyIndex(config.x_dim, 'euclidean')
    ay.load(ay_path + '.ann')
    with open(ay_path + '.txt', 'r') as f: 
        cy = np.array([line.strip() for line in f])
    
    # -- initialize

    device, kwargs = train.init(args)

    # -- model
    
    model = train.AutoGenerator(config)
    
    log_str = '{} {:.5f} {:.3e} {:.5f} {:.3e} {:d}'
    log_handle = open(os.path.join(config.out_dir, 'fate.log'), 'w') 

    names_ = []
    scores_ = []
    masks_ = []

    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    for train_pt in train_pts: 

        name = os.path.basename(train_pt).split('.')[1]

        checkpoint = torch.load(train_pt)
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(model)

        # -- evaluate 
        torch.manual_seed(0) 

        time_elapsed = config.t
        num_steps = int(np.round(time_elapsed / config.train_dt))

        scores = []
        mask = []
        pbar = tqdm.tqdm(range(len(x)), desc = "[fate:{}]".format(name))
        for i in pbar:

            # expand data point
            x_i = x[i].expand(config.ns, -1).to(device)

            # simulate forward
            for _ in range(num_steps): 
                z = torch.randn(x_i.shape[0], x_i.shape[1]) * config.train_sd
                z = z.to(device)
                x_i = model._step(x_i, dt = config.train_dt, z = z)
            x_i_ = x_i.detach().cpu().numpy()
            
            # predict
            yp = []
            for j in range(x_i_.shape[0]): 
                nn = cy[ay.get_nns_by_vector(x_i_[j], 20)]
                nn = Counter(nn).most_common(2)
                label, num = nn[0]
                if len(nn) > 1: 
                    _, num2 = nn[1]
                    if num == num2:  # deal with ties by setting it to the default class
                        label = 'Other'
                yp.append(label)
            yp = Counter(yp)

            # may want to save yp instead
            num_neu = yp['Neutrophil'] + 1 # use pseudocounts for scoring
            num_total = yp['Neutrophil'] + yp['Monocyte'] + 2
            score = num_neu / num_total
            scores.append(score)
            num_total = yp['Neutrophil'] + yp['Monocyte']
            mask.append(num_total > 0)
         
        scores = np.array(scores)
        mask = np.array(mask)

        r, pval = scipy.stats.pearsonr(y, scores)
        r_masked, pval_masked = scipy.stats.pearsonr(y[mask], scores[mask])

        log = log_str.format(name, r, pval, r_masked, pval_masked, mask.sum())
        log_handle.write(log + '\n')
        print(log)

        names_.append(name)
        scores_.append(scores)
        masks_.append(mask)

    log_handle.close()

    torch.save({
        'scores': scores_, 
        'mask': masks_, 
        'names': names_
    }, os.path.join(config.out_dir, 'fate.pt'))

def train_interpolate(args, data_path = IMPUTE_DATA_PATH): 
    a = copy.copy(args)
    
    weight = os.path.basename(a.weight_path)
    weight = weight.split('.')[0].split('-')[-1]
    a.weight = weight

    name = (
        "{weight}-"
        "{activation}_{layers}_{k_dim}-"
        "{train_dt}_{train_sd}_{train_tau}-"
        "{train_batch}_{train_clip}_{train_lr}"
    ).format(**a.__dict__)

    a.out_dir = os.path.join(args.out_dir, name, 'seed_{}'.format(a.seed)) 
    config = init_config(a)

    config.start_t = 0
    config.train_t = [2]
    config.test_t = [1]
    
    x, y, w = load_data(config)

    return x, y, w, config 


from geomloss import SamplesLoss


class OTLoss():
    
    """
    vector_a
        type: torch.Tensor
    
    vector_b 
        type: torch.Tensor
    
    weight_a
        default: None
    
    weight_b
        default: None
    
    requires_grad
        default: True
    
    Notes:
    ------
    
    Example Implementation: 
    
        a, b = torch.randint(0, 50, (10,2)), torch.randint(0, 50, (10,2))
        w = torch.ones(a.shape[0])
        w = w / w.sum()
        LF(a, b, weight_a=w, weight_b=w)
        LF(a, b)
    """
    
    def __init__(self, gpu="0"):
        
        """"""

        self.device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        self._OT_solver = SamplesLoss("sinkhorn", debias=True)
        
    def __call__(self, vector_a, vector_b, weight_a=None, weight_b=None, requires_grad=True):
        
        """"""
        
        self.vector_a = vector_a.float().to(self.device)
        self.vector_b = vector_b.float().to(self.device)
        
        if requires_grad:
            vector_a.requires_grad_()
        
        if weight_a != None and weight_b != None:
            self.weight_a = weight_a.float().to(self.device)
            self.weight_b = weight_b.float().to(self.device)

            if requires_grad:
                self.weight_a.requires_grad_()
                self.weight_b.requires_grad_()

            loss = self._OT_solver(self.weight_a, self.vector_a, self.weight_b, self.vector_b)
            return loss
        else:
            loss = self._OT_solver(self.vector_a, self.vector_b)
            return loss     

def evaluate_interpolate_model(args, config): 
   
    if not os.path.exists(config.done_log): 
        print(config.done_log, 'does not exist. Skipping.')
        return

    log_path = os.path.join(config.out_dir, 'interpolate.log')
    if os.path.exists(log_path): 
        print(log_path, 'exists. Skipping.')
        return

    x, y, w = load_data(config)
    
    # -- initialize
    device, kwargs = train.init(args) 
    model = train.AutoGenerator(config) 
    
    ot_solver = OTLoss()
#     SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur, 
#         scaling = config.sinkhorn_scaling).to(device)
    losses_xy = []
    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    for train_pt in train_pts: 
        checkpoint = torch.load(train_pt) 
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)        
        print(model) 
        name = os.path.basename(train_pt).split('.')[1]

        # -- evaluate

        def _evaluate_impute_model(t_cur): 
        
            torch.manual_seed(0)
            np.random.seed(0)

            t_prev = config.start_t
            y_prev = int(y[t_prev])
            y_cur = int(y[t_cur])
            time_elapsed = y_cur - y_prev
            num_steps = int(np.round(time_elapsed / config.train_dt))

            dat_prev = x[t_prev].to(device)
            dat_cur = x[t_cur].to(device)
            w_prev = train.get_weight(w[(y_prev, y_cur)], time_elapsed).cpu().numpy()

            x_s = []
            x_i_ = train.weighted_samp(dat_prev, args.evaluate_n, w_prev)
            for i in range(int(args.evaluate_n / config.ns)): 
                x_i = x_i_[i*config.ns:(i+1)*config.ns,]

                for _ in range(num_steps): 
                    z = torch.randn(x_i.shape[0], x_i.shape[1]) * config.train_sd
                    z = z.to(device)
                    x_i = model._step(x_i, dt = config.train_dt, z = z)

                x_s.append(x_i.detach())
            x_s = torch.cat(x_s)
            loss_xy = ot_solver(x_s.to(device), dat_cur.to(device))
            return loss_xy
 
        for t in config.train_t:
            y_ = y[t]
            loss_xy = _evaluate_impute_model(t).item()
            losses_xy.append((name, 'train', y_, loss_xy))
        try:
            for t in config.test_t: 
                y_ = y[t]
                loss_xy = _evaluate_impute_model(t).item()
                losses_xy.append((name, 'test', y_, loss_xy))
        except AttributeError:
            continue

    losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 'eval', 't', 'loss'])
    losses_xy.to_csv(log_path, sep = '\t', index = False)
    print('Wrote results to', log_path)

def evaluate_interpolate_model_baseline(args, config): 
    
    if not os.path.exists(config.done_log): 
        print(config.done_log, 'does not exist. Skipping.')
        return

    log_path = os.path.join(config.out_dir, 'baseline.log')
    if os.path.exists(log_path): 
        print(log_path, 'exists. Skipping.')
        return

    x, y, w = load_data(config)

    # -- initialize
    device, kwargs = train.init(args) 
    model = train.AutoGenerator(config) 
    
    ot_solver = SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur, 
        scaling = config.sinkhorn_scaling)

    losses_xy = []
    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    for train_pt in train_pts: 

        checkpoint = torch.load(train_pt) 
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(model) 

        name = os.path.basename(train_pt).split('.')[1]

        # -- evaluate
        torch.manual_seed(0)
        np.random.seed(0)

        t_cur = 1

        t_prev = config.start_t
        y_prev = int(y[t_prev])
        y_cur = int(y[t_cur])
        time_elapsed = y_cur - y_prev
        num_steps = int(np.round(time_elapsed / config.train_dt))

        dat_prev = x[t_prev].to(device)
        w_prev = train.get_weight(w[(y_prev, y_cur)], time_elapsed).cpu().numpy()

        x_s = []
        x_i_ = train.weighted_samp(dat_prev, args.evaluate_n, w_prev)

        for i in range(int(args.evaluate_n / config.ns)): 

            x_i = x_i_[i*config.ns:(i+1)*config.ns,]

            for _ in range(num_steps): 
                z = torch.randn(x_i.shape[0], x_i.shape[1]) * config.train_sd
                z = z.to(device)
                x_i = model._step(x_i, dt = config.train_dt, z = z)

            x_s.append(x_i.detach())

        x_s = torch.cat(x_s)

        loss_xy = [name] + [ot_solver(x_s, x[t_].to(device)).item() for t_ in range(len(x))]
        losses_xy.append(loss_xy)
 
    losses_xy = pd.DataFrame(losses_xy, columns = ['epoch'] + y)
    losses_xy.to_csv(log_path, sep = '\t', index = False)
    print('Wrote results to', log_path) 

def evaluate_interpolate_data(args, config): 
    
    x, y, w = load_data(config)

    device, kwargs = train.init(args) 

    pt = torch.load(args.evaluate_data) 
    x_i = torch.from_numpy(pt['sim_xp']).float().to(device)
    y_j = x[1].to(device)

    ot_solver = SamplesLoss("sinkhorn", p = 2, blur = config.sinkhorn_blur, 
        scaling = config.sinkhorn_scaling)
    loss_xy = ot_solver(x_i, y_j) 

    import pdb; pdb.set_trace()

def main(): 
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type = int, default = 0)
    parser.add_argument('--no-cuda', action = 'store_true')
    parser.add_argument('--device', default = 7, type = int)
    parser.add_argument('--out_dir', default = './experiments')
    # -- data options
    parser.add_argument('--data_path')
    parser.add_argument('--data_dir')
    parser.add_argument('--weight_path', default = None)
    # -- model options
    parser.add_argument('--loss', default = 'euclidean')
    parser.add_argument('--k_dim', default = 500, type = int)
    parser.add_argument('--activation', default = 'softplus')
    parser.add_argument('--layers', default = 1, type = int)
    # -- pretrain options
    parser.add_argument('--pretrain_lr', default = 1e-9, type = float)
    parser.add_argument('--pretrain_epochs', default = 500, type = int) 
    # -- train options 
    parser.add_argument('--train_epochs', default = 5000, type = int)
    parser.add_argument('--train_lr', default = 0.01, type = float)
    parser.add_argument('--train_dt', default = 0.1, type = float)
    parser.add_argument('--train_sd', default = 0.5, type = float)
    parser.add_argument('--train_tau', default = 0, type = float)
    parser.add_argument('--train_batch', default = 0.1, type = float)
    parser.add_argument('--train_clip', default = 0.25, type = float)
    parser.add_argument('--save', default = 100, type = int)
    # -- test options
    parser.add_argument('--evaluate_n', default = 10000, type = int)
    parser.add_argument('--evaluate_data')
    parser.add_argument('--evaluate-baseline', action = 'store_true')
    # -- run options
    parser.add_argument('--task', default = 'fate')
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--evaluate')
    parser.add_argument('--config')
    args = parser.parse_args() 

    if args.task == 'fate': 

        if args.train: 

            args.pretrain = True
            args.train = True

            train.run(args, train_fate)

        if args.evaluate == 'model': 

            if args.config:
                config = SimpleNamespace(**torch.load(args.config))
                evaluate_fate(args, config) 
            else: 
                print('Please provide a config file')

    elif args.task == 'interpolate': 
        
            args.gpu = 0
        
            if args.train: 

                args.pretrain = True
                args.train = True

                config = train.run(args, train_interpolate) 

            elif args.evaluate: 

                if args.evaluate == 'model':
                    print("evaluating model...")
                    evaluate = evaluate_interpolate_model
                elif args.evaluate == 'data': 
                    evaluate =  evaluate_interpolate_data
                elif args.evaluate == 'baseline': 
                    evaluate = evaluate_interpolate_model_baseline
                else:
                    raise NotImplementedError

                if args.config:
                    config = SimpleNamespace(**torch.load(args.config))
                else:
                    print("Please provide a config file")

                evaluate(args, config)

            


if __name__ == '__main__':
    main()
