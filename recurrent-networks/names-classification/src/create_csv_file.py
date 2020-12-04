import pandas as pd 
import os 
import numpy as np 

from utils import findFiles, readLines, letterToTensor, lineToTensor
from tqdm import tqdm

data_dir = '../../../data/data'
csv_file_name = 'train.csv'




if __name__ == '__main__':
    category_lines = {}
    all_categories = []
    df = pd.DataFrame()
    #file_names = os.listdir(os.path.join(data_dir, 'names'))
    for filename in tqdm(findFiles(os.path.join(data_dir, 'names/*.txt')), desc='Creating csv file'):
        category = os.path.splitext(os.path.basename(filename))[0]
        #all_categories.append(category)
        lines = readLines(filename)
        #category_lines[category] = lines
        categories = [category for _ in range(len(lines))]
        for (line, cat) in zip(lines, categories):
            df = df.append({
                'name' : line,
                'target': cat
            }, ignore_index=True)

    # shuffle datasets
    df = df.sample(frac=1).reset_index(drop=True)
    # save dataframe as csv file
    df.to_csv(os.path.join(data_dir, csv_file_name), index=False)
    print(f"[INFO] {csv_file_name} saved to {data_dir}")
