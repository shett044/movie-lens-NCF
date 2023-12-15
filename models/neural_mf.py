import torch
from torch import nn
from typing import List
from models import gmf, mlp
class NeuMF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, num_layers, pretrained_models = None) -> None:
        super().__init__()
        self.num_factors = num_factors
        self.num_users = num_users
        self.num_items = num_items
        
        self.GMF = gmf.GMF(num_factors, num_users, num_items, get_embed=True)
        self.MLP = mlp.MLP(num_factors, num_users, num_items, num_layers, get_embed= True)
        self.fc = nn.Linear(2*num_factors, 1)
        self.sig = nn.Sigmoid()
        self.is_pretrain=bool(pretrained_models)
        self._init_weights()
        if pretrained_models:
            self._load_pretrained(pretrained_models)
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
    
    def _load_pretrained(self, pretrained_models: List[nn.Module]):
        self.GMF.user_embed.weight.data = pretrained_models[0].user_embed.weight.data
        self.GMF.item_embed.weight.data = pretrained_models[0].item_embed.weight.data

        self.MLP.user_embed.weight.data = pretrained_models[1].user_embed.weight.data
        self.MLP.item_embed.weight.data = pretrained_models[1].item_embed.weight.data
        for idx in range(len(self.MLP.fcLayers)):
            if isinstance(self.MLP.fcLayers[idx], nn.Linear):
                self.MLP.fcLayers[idx].weight.data = pretrained_models[1].fcLayers[idx].weight.data


        wts = torch.cat([pm.fc.weight for pm in pretrained_models], dim = -1)
        bias = sum([pm.fc.bias for pm in pretrained_models])
        self.fc.weight.data.copy_(  0.5 * wts )
        self.fc.bias.data.copy_( 0.5 * bias)
    
    def forward(self, user, item):
        embeds = torch.cat([self.GMF(user, item), self.MLP(user, item)], dim = -1)
        res = self.fc(embeds)
        res = self.sig(res).view(-1)
        return res




