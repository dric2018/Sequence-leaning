# Tutorial from : https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import sys
import unicodedata
import string
import torch
import random
import pandas as pd


data_dir = '../../../data/data/'


def findFiles(path): return glob.glob(path)

def get_max_len():
    max_len = 0
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    names = df.name.tolist()

    for name in names:
        if len(name)>max_len:
            big_name = name
            max_len = len(name)

    return big_name, max_len


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
_, max_len = get_max_len()

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <max_length x 1 x n_letters> to make sure all tensors have same size
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(max_len, 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor



if __name__ == '__main__':
    for filename in findFiles(os.path.join(data_dir, 'names/*.txt')):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    #print(findFiles(os.path.join(data_dir, 'names/*.txt')))
    #print(f'> {n_categories} found')
    #print(category_lines['Italian'][:5])
    #print(unicodeToAscii('Ślusàrski'))
    #print(letterToTensor('J'))
    #print(lineToTensor('Jones').size())
    print(n_letters)