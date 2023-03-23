import os

import pandas as pd

train = pd.read_pickle(os.path.join('dataset/ml-1m/train.df'))
test = pd.read_pickle(os.path.join('dataset/ml-1m/test.df'))
x = 1