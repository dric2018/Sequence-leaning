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

#### create arguments parser

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default=os.path.join(data_dir, 'train.csv'), type=str, help='path to csv file to use as dataset')
parser.add_argument("--lr", default=0.005, type=float, help='Learning rate to use for training')
parser.add_argument("--num_epochs", default=100000, type=int, help='Number of training epochs')
parser.add_argument("--check_every", default=1000, type=int, help='Number of epochs to check metrics')



if __name__ == '__main__':
    args = parser.parse_args()
    train = pd.read_csv(args.dataset_path)
    classes = train.target.unique().tolist()
    dataset = NamesDataset(csv_file=train)


    dataloader = DataLoader(dataset=dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=os.cpu_count())


    model = NameClassifier(lr=args.lr)

    ckpt_callback = pl.callbacks.ModelCheckpoint(filename=f'names_classifier', 
                                               dirpath='../../../models', 
                                               monitor='train_loss', 
                                               mode='min')  

    trainer = Trainer(max_epochs=args.num_epochs, callbacks=[ckpt_callback])
                                               
    trainer.fit(model, dataloader)


