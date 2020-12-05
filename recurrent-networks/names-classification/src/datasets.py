import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from pytorch_lightning import seed_everything
import pandas as pd
import argparse
import os 
from utils import lineToTensor, letterToIndex, letterToTensor



### CONSTANTS
data_dir = '../../../data/data/'


#### create arguments parser

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path", default=os.path.join(data_dir, 'train.csv'), type=str, help='path to csv file to use as dataset')
parser.add_argument("--batch_size", default=32, type=int, help='Training batch size')


class NamesDataset(Dataset):
    def __init__(self, csv_file, num_classes=18, task='train', **kwargs):
        super(NamesDataset, self).__init__()
        self.df = csv_file
        self.num_classes = num_classes
        self.task = task
        self.class_dict = {c:idx for idx, c in enumerate(self.df.target.unique().tolist())}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        name = self.df.iloc[index][0]

        data = {
            'name' : name,
            'x' : lineToTensor(name)
        }

        if self.task == 'train':
            target = self.df.iloc[index][1]

            data.update({
                'y' : torch.tensor(self.class_dict[target], dtype=torch.long)
            })


        return data



def my_collate(batch):

    """
    from : https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2
    """
    names = [item['name'] for item in batch]
    data = [item['x'] for item in batch]
    target = [item['y'] for item in batch]

    # convert lists to tensors
    data = torch.cat(data)
    target = torch.LongTensor(target)
    sample = [names, data, target]

    return sample



if __name__ == '__main__':
    args = parser.parse_args()
    train = pd.read_csv(args.dataset_path)
    classes = train.target.tolist()
    dataset = NamesDataset(csv_file=train)


    dataloader = DataLoader(dataset=dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=os.cpu_count(), collate_fn=my_collate)

    for data in dataloader:
        names, xs, ys = data
        print(xs.shape)
        print(ys.shape)

        print(f"name : {names[0]} | class : {classes[ys[0].item()]}")

        break