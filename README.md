## NCFML (NCF with MovieLens 100 K in torch)

### Dataset
This repository is about Neural Collaborative Filtering with MovieLens in torch based on the Neural Collaborative Filtering [paper](https://arxiv.org/abs/1708.05031).

Dataset is Implict Feedback, If there is interaction between user and item, then target value will be 1.So if there is rating value between user and movie, then target value is 1, otherwise 0. 

For negative sampling, ratio between positive feedback and negative feedback is 1:4 in trainset, and 1:99 in testset. (these ratios are same as author's code [@hexiangnan](https://github.com/hexiangnan/neural_collaborative_filtering))

## Parameters

| Parameter | type | default | description|
| --- | --- | --- | --- |
| "--epochs" | type=int | default=20 | help = "number of epochs" |
|"--batch_size" | type=int | default=256 | help = "Batch size"|
|"--lr" | type=float | default=.05 | help = "Learning rate"|
|"--factors" | type=int | default=8 | help = "number of factors"|
|"--layers" | type=list | default=[64 ,32 , 16] | help = "Size of hidden layers in list"|
|"--save-model" | type=str | default='y' | help = "Save model y/n"|
|"--use-pretrain" | type=str | default='y' | help = "Use Pretrains y/n"|
|"--save-ds" | type=str | default='y' | help = "Use Saved dataset y/n"|
|"--top-k" | type=int | default=10 | help = "Top k to evaluate NDCG | MAP"|
|"--model" | type=str | default="NeuMF" | help = "select among the following model |[MLP | GMF | NeuMF]""|



### Quick start
```python
python main.py --epochs 30 --batch_size 256 --factors 8 --model NeuMF --top-k 10  --layer 64 32 16 --use_pretrain False

```

### Development enviroment

- OS: Linux
- IDE: VSCode
- GPU: NVIDIA




### Example of command line

- save GMF
  ```python
  python main.py --epochs 20 --batch 256 --factors 8 --model GMF --top-k 10
  --layer 64 32 16 --download True --save True

  ```
- save MLP

  ```python
  python main.py --epochs 20 --batch 256 --factors 8 --model MLP --top-k 10
   --layer 64 32 16 --download False --save True

  ```
- use pre-trained model
  ```python
  python main.py --epochs 20 --batch 256 --factors 8 --model NeuMF  --top-k 10
   --layer 64 32 16  --use_pretrain True
  ```

### Example of logfile

```
2023-12-14 22:48:33 - INFO - DEVICE = device(type='cuda')
2023-12-14 22:48:33 - INFO - Current parameters
2023-12-14 22:48:33 - INFO - {'epochs': 20, 'batch_size': 256, 'lr': 0.05, 'factors': 8, 'layers': [64, 32, 16], 'save_model': 'y', 'use_pretrain': 'y', 'save_ds': 'y', 'top_k': 10, 'model': 'NeuMF'}
2023-12-14 22:48:33 - INFO - Reading CSV completed
2023-12-14 22:48:33 - INFO - Data loaded in : 0.24289488792419434 sec
2023-12-14 22:49:03 - INFO - epoch= 1 HR =0.62 NDCG = 0.37 Time = 27.75 sec
2023-12-14 22:49:29 - INFO - epoch= 2 HR =0.65 NDCG = 0.40 Time = 26.65 sec
2023-12-14 22:50:00 - INFO - epoch= 3 HR =0.69 NDCG = 0.44 Time = 30.97 sec
2023-12-14 22:50:27 - INFO - epoch= 4 HR =0.68 NDCG = 0.43 Time = 26.81 sec
2023-12-14 22:50:58 - INFO - epoch= 5 HR =0.71 NDCG = 0.45 Time = 30.63 sec
2023-12-14 22:51:24 - INFO - epoch= 6 HR =0.72 NDCG = 0.46 Time = 26.63 sec
2023-12-14 22:51:50 - INFO - epoch= 7 HR =0.71 NDCG = 0.45 Time = 26.02 sec
2023-12-14 22:52:16 - INFO - epoch= 8 HR =0.71 NDCG = 0.45 Time = 25.89 sec
2023-12-14 22:52:43 - INFO - epoch= 9 HR =0.72 NDCG = 0.47 Time = 26.91 sec
2023-12-14 22:53:11 - INFO - epoch= 10 HR =0.71 NDCG = 0.46 Time = 27.74 sec
2023-12-14 22:53:39 - INFO - epoch= 11 HR =0.72 NDCG = 0.47 Time = 28.25 sec
2023-12-14 22:54:05 - INFO - epoch= 12 HR =0.69 NDCG = 0.44 Time = 26.00 sec
2023-12-14 22:54:32 - INFO - epoch= 13 HR =0.72 NDCG = 0.46 Time = 27.19 sec
2023-12-14 22:54:59 - INFO - epoch= 14 HR =0.71 NDCG = 0.44 Time = 26.88 sec
2023-12-14 22:55:26 - INFO - epoch= 15 HR =0.71 NDCG = 0.45 Time = 26.86 sec
2023-12-14 22:55:53 - INFO - epoch= 16 HR =0.72 NDCG = 0.46 Time = 26.99 sec
2023-12-14 22:56:20 - INFO - epoch= 17 HR =0.72 NDCG = 0.46 Time = 26.45 sec
2023-12-14 22:56:47 - INFO - epoch= 18 HR =0.72 NDCG = 0.46 Time = 27.47 sec
2023-12-14 22:57:13 - INFO - epoch= 19 HR =0.71 NDCG = 0.45 Time = 25.76 sec
2023-12-14 22:57:39 - INFO - epoch= 20 HR =0.72 NDCG = 0.46 Time = 26.30 sec
```

## Reference
1. [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
2. Official [code](https://github.com/hexiangnan/neural_collaborative_filtering) from author
