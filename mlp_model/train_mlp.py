import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import r2_score

from train_utils import *

device = get_device_map()
print(device)

csv_dir = 'C:\\Users\\user\\Desktop\\고려대\\3학년\\데과\\공모전\\DataScience\\종합\\normalized_score'
csvs = os.listdir(csv_dir)
csvs = [os.path.join(csv_dir, filename) for filename in csvs]

train_year =  ['2017', '2018', '2020', '2021', '2022', '2023']
test_year = ['2024', '2019']

train_split, test_split = loader_by_year(train_year, test_year, csvs)

#print(train_split)
input_dim = len(train_split[train_year[0]][0]['boy'])

model = nn.Sequential(
    nn.Linear(input_dim*2, 500),
    nn.ReLU(),
    nn.Linear(500, 500),
    nn.ReLU(),
    nn.Linear(500, input_dim),
).to(device)

loss_mean = 0
loss_naive = 0
loss_trained = 0
r2 = 0

loss_list = []
#optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.3)
criterion = nn.KLDivLoss(reduction='batchmean')
epochs = 800

target = 'full'
month = 6
#print(test_split)

model.eval()
for year, i in zip(test_year, range(1, len(test_year)+1)):
    inputs = torch.tensor(list(test_split[year][-1][target] + test_split[year][-2][target])).to(device)
    gt = torch.tensor(test_split[year][0][target]).to(device)
    logits = model(inputs)
    logits = logits.unsqueeze(0)
    print(logits.shape)
    predicted_probs = F.log_softmax(logits, dim=1)
    loss_naive += criterion(predicted_probs, gt)
    

print('naive loss: ', loss_naive/len(test_year))

model.train()
for _ in range(epochs):
    for year, i in zip(train_year, range(1, len(train_year)+1)):        
        inputs = torch.tensor(list(train_split[year][-1][target] + train_split[year][-2][target])).to(device)
        gt = torch.tensor(train_split[year][0][target]).to(device)

        logits = model(inputs)
        logits = logits.unsqueeze(0)
        predicted_probs = F.log_softmax(logits, dim=1)
        loss = criterion(predicted_probs, gt)

        loss.backward()
        optimizer.step()
        loss_mean += loss.item()
    loss_list.append(float(loss_mean/(len(train_year)*(_+1))))
    print(f'loss at epoch {_}:', loss_mean/(len(train_year)*(_+1)))

model.eval()
print(test_year)
for year, i in zip(test_year, range(1, len(test_year)+1)):
    inputs = torch.tensor(list(test_split[year][-1][target] + test_split[year][-2][target])).to(device)
    gt = torch.tensor(test_split[year][0][target]).to(device)
    logits = model(inputs)
    logits = logits.unsqueeze(0)
    predicted_probs = F.log_softmax(logits, dim=1)
    loss_trained += criterion(predicted_probs, gt)
    predicted_probs_cpu = predicted_probs.reshape(-1).detach().cpu().numpy()
    gt_cpu = gt.cpu().numpy()
    
    r2 += r2_score(predicted_probs_cpu, gt_cpu)
    print(r2)

print('loss after training: ', loss_trained.item()/len(test_year))
print('r2 score after training: ', r2/len(test_year))
print('naive loss: ', loss_naive.item()/len(test_year))
draw_graph(loss_list)





