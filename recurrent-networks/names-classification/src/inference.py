# I will be using pytorch lightning for this tutorial

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from pytorch_lightning import Trainer, seed_everything, loggers, callbacks
from models import NameClassifier
# import dataset stuff
from datasets import NamesDataset, my_collate
from utils import lineToTensor
import random 
import argparse
import os
import pandas as pd
import gc

data_dir = '../../../data/data/'
model_dir = '../../../models/'

#### create arguments parser

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default=os.path.join(data_dir, 'train.csv'), type=str, help='path to csv file to use as dataset')

# Just return an output given a line
def evaluate(line_tensor, model):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    return output.argmax(1)

if __name__ == '__main__':
    args = parser.parse_args()
    train = pd.read_csv(args.dataset_path)
    classes = train.target.unique().tolist()

    net = NameClassifier()
    try:
        net.load_from_checkpoint(os.path.join(model_dir, 'names_classifier.ckpt'))
    except:
        net.load_from_checkpoint(os.path.join(model_dir, 'names_classifier-v0.ckpt'))
    
    net.eval()

    while(1):
        name_line = input('[🤗 Classifier - running] Enter the name : \n>>> ')
        try:
            if name_line in ['q', 'quit']:
                print('🤗 Bye !')
                break
            else:
                name_tensor = lineToTensor(name_line)
            
                pred = evaluate(name_tensor, net)

                c = classes[pred]

                print(f"[Classifier] >>> {name_line} seems to be a {c} name i think...\n")
        
        except Exception as e:
            print(f"[Error] {e}")

