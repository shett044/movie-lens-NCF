import torch
from torch import nn
import logging


class MLP(nn.Module):
    def __init__(self,  num_factors, num_users, num_items, layers, get_embed = False) -> None:
        super().__init__()
        self.user_embed = nn.Embedding(num_users, layers[0]//2)
        self.item_embed = nn.Embedding(num_items, layers[0]//2)
        layers_fc = []
        for l in layers:
            layers_fc.append(nn.Linear(l, l//2))
            layers_fc.append(nn.BatchNorm1d(l//2))
            layers_fc.append(nn.ReLU())
        
        self.fcLayers = nn.Sequential(*layers_fc)
        self.fc = nn.Linear(num_factors, 1)
        self.sigmoid = nn.Sigmoid()
        self.get_embed = get_embed
    
    def forward(self, user, item):
        user_em = self.user_embed(user)
        item_em = self.item_embed(item)
        embeds = torch.cat([user_em, item_em], dim = -1)
        

        output = self.fcLayers(embeds)
        if self.get_embed:
            return output
        output = self.fc(output)

        output = self.sigmoid(output).view(-1)
        
        return output