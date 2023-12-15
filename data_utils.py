import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Tuple
from tqdm import tqdm

class MovieLens(Dataset):
    def __init__(self, df: pd.DataFrame, user_item_matrix:pd.DataFrame, neg_sample = 4, device = 'cpu'):
        super(MovieLens, self).__init__()
        self.df = df.eval('rating = 1').drop('ts',1)
        self.user_item_map = user_item_matrix
        # .apply(lambda x: set(x[x==0].index), axis=1)
        self.neg_sample = neg_sample
        self.device = device
        self.users, self.items, self.labels = self._neg_sampling()
        self.users.to(self.device)
        self.items.to(self.device)
        self.labels.to(self.device)

    def _neg_sampling(self):
        data = []
        for i, row in tqdm(self.df.iterrows(), total = self.df.shape[0]):
            u = row['userId']
            neg_df =  pd.DataFrame({"movieId":np.random.choice(self.user_item_map[u], size= self.neg_sample)}).assign(userId = u, rating = 0)
            data.append(self.df.loc[[i]])
            data.append(neg_df)
        res = data
        data = pd.concat(data)
        return torch.LongTensor(data['userId'].values), torch.LongTensor(data['movieId'].values) ,torch.Tensor(data['rating'].values)
                
            
    def __len__(self):
        return len(self.users)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[index], self.items[index], self.labels[index]



