import scipy.sparse as sp
import pandas as pd
path = '../dataset/tmall/s_pre_adj_mat.npz'
# df = pd.read_csv(path, delimiter='\t')
pre_adj_mat = sp.load_npz(path)
x = 1
