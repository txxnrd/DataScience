import numpy as np
import pandas as pd
import csv
import torch
import matplotlib.pyplot as plt


def get_device_map() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def csv_to_dist(csv_path):                          #### returns normalized boy, girl, full dist and students
    boy_dist = dict()
    girl_dist = dict()
    full_dist = dict()
    boy_sum, girl_sum = 0, 0
    with open(csv_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)    
        for row in reader:
            boy_dist[row['normalized_score']] = int(row['male'])
            girl_dist[row['normalized_score']] = int(row['female'])
            full_dist[row['normalized_score']] = int(row['male']) + int(row['female'])
            
            boy_sum += int(row['male'])
            girl_sum += int(row['female'])

    boy_dist = [value/boy_sum for _, value in boy_dist.items()]
    girl_dist = [value/boy_sum for _, value in girl_dist.items()]
    full_dist = [value/boy_sum for _, value in full_dist.items()] 

    return boy_dist, girl_dist, full_dist



def loader_by_year(train_year, test_year, csvs):
    train_dict = dict()
    test_dict = dict()
    train_dict = {year: [] for year in train_year}
    test_dict = {year: [] for year in test_year}

    for path in csvs:
        #print(path.split('_'))
        boy, girl, full = csv_to_dist(path)
        if path.split("_")[2] in train_year:
            train_dict[path.split("_")[2]].append({'boy': boy, 'girl': girl, 'full': full})
        elif  path.split("_")[2] in test_year:
            test_dict[path.split("_")[2]].append({'boy': boy, 'girl': girl, 'full': full})
    return train_dict, test_dict

    
def draw_graph(loss_list):
    epochs = len(loss_list)
    x_axis = range(len(loss_list))

    plt.plot(x_axis, loss_list)
    plt.ylim(0, 2.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()