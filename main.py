import torch
import argparse

import numpy as np
import pandas as pd
from models.gmf import GMF
from models.mlp import MLP
from models.neural_mf import NeuMF
from torch.utils.data import DataLoader
from torch.optim import Adam
import data_utils as ut_data
from trainer import Trainer
import evaluators as eval
import time

try:
    from argparser import args
except:
    from types import SimpleNamespace
    d  = {'epochs': 20, 'batch_size': 256, 'lr': 0.001, 'factors': 8, 'layers': [64, 32, 16], 'save_model': 'y', 'use_pretrain': 'y', 'top_k': 10, 'model': 'GMF', 'save_ds':'n'}
    args = SimpleNamespace(**d)
    
from boilerplate import *

logInfo("Current parameters")
logInfo(args.__dict__)

use_pretrain  = False if args.use_pretrain == 'n' else True
save_model  = False if args.save_model == 'n' else True
save_ds  = False if args.save_ds == 'n' else True

TRAIN_NEG_SAMPLE = 4
TEST_NEG_SAMPLE = 99
"""
Load Data
"""
ts_st_data = time.time()
DIR_DATA = Path("ml-100k")
tot_df = pd.read_csv(DIR_DATA.joinpath('u.data'), sep='\t',names=['userId','movieId', 'rating', 'ts'])
num_u, num_i = tot_df.userId.nunique() + 1, tot_df.movieId.nunique() + 1
logInfo("Reading CSV completed")
if not save_ds:
    logInfo("Processing Data set")
    movie_set = np.unique(tot_df.movieId)
    user_item_matrix = tot_df.groupby(['userId'])['movieId'].apply(lambda x: np.setdiff1d(movie_set, x.values))

    train_df, val_df = ut.split_train_val_test(tot_df, val_frac=.1)
    train_ds = ut_data.MovieLens(train_df, user_item_matrix, neg_sample=TRAIN_NEG_SAMPLE, device = DEVICE)
    test_ds = ut_data.MovieLens(val_df, user_item_matrix, neg_sample=TEST_NEG_SAMPLE, device = DEVICE)
    torch.save(train_ds, DIR_DATA.joinpath('train.pt'))
    torch.save(test_ds, DIR_DATA.joinpath('test.pt'))
else:
    train_ds = torch.load(DIR_DATA.joinpath('train.pt'))
    test_ds = torch.load(DIR_DATA.joinpath('test.pt'))

ts_en_data = time.time()
logInfo(f"Data loaded in : {ts_en_data - ts_st_data} sec")

train_loader = DataLoader(train_ds, args.batch_size, shuffle = True, num_workers=0, drop_last=True)
test_loader = DataLoader(test_ds, TEST_NEG_SAMPLE + 1, shuffle = False,num_workers=0, drop_last=True)

"""
Load Model
"""

if args.model == "GMF":
    model = GMF(args.factors, num_u, num_i)
    ut.init_hidden_xavier(model)
elif args.model == "MLP":
    model = MLP( args.factors, num_u, num_i, args.layers)
    ut.init_hidden_xavier(model)
else:
    pretrained_models = None
    if use_pretrain:
        pretrained_models = [torch.load('results/best_model/gmf_pretrained.pth' ) , torch.load('results/best_model/mlp_pretrained.pth' ) ,]
    model = NeuMF( args.factors, num_u, num_i, args.layers, pretrained_models)
    if not use_pretrain:
        ut.init_hidden_xavier(model)
    
model.to(DEVICE)


"""
Load Torch essentials
"""
optim = Adam(model.parameters(), lr = args.lr)
loss = torch.nn.BCELoss()




if __name__=='__main__':
    train = Trainer(model, optim, args, train_loader, test_loader, loss, DEVICE, logInfo)
    # measuring time
    ts_st_model = time.time()
    train.train()
    if save_model:
        torch.save(model,DIR_PRETRAIN.joinpath(args.model + ".pth"))
    ts_en_model = time.time()

    print(f'training time:{ts_en_model-ts_st_model:.5f} sec')
    HR,NDCG = eval.evaluate(model,test_loader,topk=args.top_k,device=DEVICE)
    print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
