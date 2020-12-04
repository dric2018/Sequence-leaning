# I will be using pytorch lightning for this tutorial

import torch
import torch.nn as nn 
import pytorch_lightning as pl 
from pytorch_lightning import Trainer, seed_everything, loggers, callbacks

# import dataset stuff
from utils import letterToTensor, lineToTensor, categoryFromOutput, randomChoice, randomTrainingExample

import random 

# Network definition 
# see the model graph design in imgs/name_classifier.png
class NameClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=5e-3):
        super(NameClassifier, self).__init__()

        self.hidden_size = hidden_size

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
        self.hidden = self.initHidden()
        x, y = batch

        out, hidden = self(x[batch_idx], self.hidden)

        loss = get_loss(logits=out, targets=y) 



    def validation_step(self, batch, batch_idx):
        pass

    def get_loss(self, logits, targets):
        return nn.NLLLoss()(logits, targets)

    def get_acc(self, logits, targets):
        pred_class, pred_class_i = categoryFromOutput(logits)



if __name__ == '__main__':
    n_letters = 57
    n_hidden = 128
    n_categories = 18
    net = NameClassifier(input_size=n_letters, hidden_size=n_hidden, output_size=n_categories)
    #print(net)

    #input_ = letterToTensor('A')
    #hidden = torch.zeros(1, n_hidden)
    #output, next_hidden = net(input_, hidden)
    #print(f'[INFO] output shape: {output.shape} | hidden_state shape: {hidden.shape}')

    #del hidden
    # input_1 = lineToTensor('Albert')
    # hidden = torch.zeros(1, n_hidden)

    # output, next_hidden = net(input_1[0], hidden)
    # print(f'[INFO] output shape: {output.shape} | hidden_state shape: {next_hidden.shape}')
    # print(categoryFromOutput(output))

    for i in range(5):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)


