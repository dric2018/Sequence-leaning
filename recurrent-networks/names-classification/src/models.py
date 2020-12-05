# I will be using pytorch lightning for this tutorial

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import pytorch_lightning as pl 
from pytorch_lightning import Trainer, seed_everything, loggers, callbacks

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




# Network definition 
# see the model graph design in imgs/name_classifier.png
class NameClassifier(pl.LightningModule):
    def __init__(self, input_size=57, hidden_size=128, output_size=18, lr=5e-3):
        super(NameClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.lr = lr

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x, hidden):
        combined = torch.cat((x, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

        

    def training_step(self, batch, batch_idx):
        name, x, y = batch.values()

        hidden = self.initHidden()
        for i in range(x.size()[1]):
            out, hidden = self(x.squeeze(0)[i], hidden)

        loss = self.get_loss(logits=out, targets=y) 
        guess = self.get_acc(logits=out, targets=y)

        self.log("prediction", guess, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)


        return {"loss":loss, "train_acc":guess}

    
    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(), lr=self.lr)

        return opt


    def get_loss(self, logits, targets):
        return nn.NLLLoss()(logits, targets)

    def get_acc(self, logits, targets):

        acc = (logits.argmax(1) == targets).int()

        if acc == 0:
            guess = '✗ incorrect'
        else:
            guess = '✓ correct'

        return acc



if __name__ == '__main__':

    args = parser.parse_args()
    train = pd.read_csv(args.dataset_path)
    classes = train.target.unique().tolist()
    dataset = NamesDataset(csv_file=train)


    dataloader = DataLoader(dataset=dataset, 
                            batch_size=1, 
                            shuffle=True, 
                            num_workers=os.cpu_count())

    net = NameClassifier()
    #print(net)

    for data in dataloader:
        name, x, y = data.values()

        hidden = net.initHidden()
        for i in range(x.size()[1]):
            out, hidden = net(x.squeeze(0)[i], hidden)

        loss = net.get_loss(out, y)
        acc = net.get_acc(out, y)

        print(loss)
        print(acc)
        break

    gc.collect()


    




