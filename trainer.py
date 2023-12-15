import torch
from evaluators import evaluate
import time 
from datetime import datetime

class Trainer:
    def __init__(self, model:torch.nn.Module, optimizer:torch.optim.Optimizer, model_args:object, dataloader: torch.utils.data.DataLoader, testobj, criterion: torch.nn.Module, device = 'cpu', logger = print ):
        self.model = model
        self.optim = optimizer
        self.model_args = model_args
        self.dataloader = dataloader
        self.test = testobj
        self.criterion = criterion
        self.device = device
        self.log = logger
        self.MODEL_DIR = "results/best_model/"
        model_name = model_args.model if 'model' in model_args else 'best_model'
        self.MODEL_FILE = model_name + datetime.now().strftime('%Y-%m-%d_%H-%M') + ".pth"

    def train(self):
        best_ndcg = 100
        self.model.to(self.device)
        for epoch in range(1, self.model_args.epochs + 1) :
            stime = time.time()
            for user, item, label in self.dataloader:
                user, item, label = user.to(self.device), item.to(self.device), label.to(self.device)
                self.optim.zero_grad()
                pred = self.model(user, item)
                cost = self.criterion(pred, label)
                cost.backward()
                self.optim.step()
            HR, NDCG = evaluate(self.model,self.test,10, self.device)
            if NDCG < best_ndcg:
                torch.save(self.model, self.MODEL_DIR + self.MODEL_FILE)
                best_ndcg = NDCG
                # if self.model_args.model == "NeuMF" and self.model.is_pretrain:
                #     torch.save(self.model.GMF, 'results/best_model/gmf_pretrained.pth')
                #     torch.save(self.model.MLP, 'results/best_model/mlp_pretrained.pth')

            etime = time.time()
            self.log(f"{epoch= } {HR =:.2f} {NDCG = :.2f} Time = {etime - stime:.2f} sec")
        return 



