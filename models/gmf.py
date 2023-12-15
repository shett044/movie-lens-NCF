import torch
from torch import nn
import logging
class GMF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, get_embed = False):
        super(GMF, self).__init__()
        self.num_factors = num_factors
        self.nu = num_users
        self.ni = num_items
        self.user_embed = nn.Embedding(num_users, num_factors)
        self.item_embed = nn.Embedding(num_items, num_factors)
        self.fc = nn.Linear(num_factors, 1)
        self.sigmoid = nn.Sigmoid()
        self.get_embed = get_embed

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.item_embed.weight)
        nn.init.xavier_normal_(self.fc.weight)


    def forward(self, users, items):
        # logging.info(f"Users : {max(users)}, Item: {max(items)}")
        U = self.user_embed(users)
        I = self.item_embed(items)
        embed = U * I
        if self.get_embed:
            return embed

        res = self.sigmoid(self.fc(embed))
        return res.view(-1)