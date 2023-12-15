import numpy as np
from typing import Tuple
import torch

def hit(actual_item: int, preds:set)->int:
    return 1 if actual_item in preds else 0

def ndcg(actual_item:int, preds: list) -> float:
    try:
        rank = preds.index(actual_item) + 1 
        return np.reciprocal(np.log2(1 + rank))
    except:
        return 0.0

def evaluate(model:torch.nn.Module, test_loader: torch.utils.data.DataLoader, topk:int, device = 'cpu')-> Tuple[float, float]:
    HR, NDCG = [], []
    model.eval()
    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)
            predictions = model(user, item)
            _, idx = torch.topk(predictions, topk)
            recommends = torch.take(item, idx).cpu().numpy().tolist()
            act_item = item[0].item()
            HR.append(hit(act_item, set(recommends)))
            NDCG.append(ndcg(act_item, recommends))
    return np.mean(HR), np.mean(NDCG)
