import torch
from recbole.model.abstract_recommender import SequentialRecommender
import scipy.sparse as sp
from time import time
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix, coo_matrix

class GraphDataCollector(SequentialRecommender):
    def __init__(self, config, dataset, train_data):
        super().__init__(config, dataset)
        self.Graph = None
        self.user_num = dataset.num(self.USER_ID)
        self.item_num = dataset.num(self.ITEM_ID)
        self.behavior_num = dataset.num(self.BEHAVIOR_ID)
        self.item2Idx = dataset.field2token_id['item_id']
        self.user2Idx = dataset.field2token_id['item_id']
        self.userSet = train_data.dataset.inter_feat.interaction['user_id']
        self.item_seq = train_data.dataset.inter_feat.interaction['item_id_list']
        self.behavior_seq = train_data.dataset.inter_feat.interaction['behavior_type_list']

        self.userIdx2sequence = dict(zip(self.userSet, self.item_seq))
        # self.userItemSet = dict(zip(self.userSet, set(self.item_id_list)))
        self.itemSeq2user = {}
        self.user_dist = [0 for _ in range(self.user_num)]
        self.item_dist = [0 for _ in range(self.item_num)]
        self.getSparseGraph()

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.data_path + f'/s_pre_adj_mat_{self.hop_num}.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("start generating UserUserNet")
                
                rating_mat = torch.zeros((self.user_num,self.item_num))
                for user1, item_seq1 in self.userIdx2sequence.items():
                    rating_mat[user1][item_seq1] = 1
                
                s_uu = time()
                UserUserNet, UserItemNet, ItemItemNet = None, None, None
                rows, cols, weights = [], [], []
                c = 0

                for user1 in range(1,rating_mat.size()[0]):
                    print(user1)
                    for user2 in range(1,rating_mat.size()[0]):
                        neighbors_num = torch.sum(rating_mat[user1]*rating_mat[user2])
                        if neighbors_num == 0:
                                continue
                        weight =  neighbors_num / (torch.sum(rating_mat[user1]) + torch.sum(rating_mat[user2]) - neighbors_num)
                        rows.append(user1)
                        cols.append(user2)
                        weights.append(weight)
                        rows.append(user2)
                        cols.append(user1)
                        weights.append(weight)

                # # build UserUserNet
                
                # for user1, item_seq1 in self.userIdx2sequence.items():
                #     # c += 1
                #     # if c%1000 == 0:
                #     print(user1)
                #     for user2, item_seq2 in self.userIdx2sequence.items():
                        
                #         if user2 < user1 or user2 == user1:
                #             continue
                #         else:
                #             neighbors = set(item_seq1) & set(item_seq2)
                #             if len(neighbors) == 0:
                #                 continue
                #             print("neighbors")
                #             weight = len(neighbors) / len(set(item_seq1) | set(item_seq2))
                #             rows.append(user1)
                #             cols.append(user2)
                #             weights.append(weight)
                #             rows.append(user2)
                #             cols.append(user1)
                #             weights.append(weight)

                UserUserNet = csr_matrix((weights, (rows, cols)), shape=(self.user_num, self.user_num))
                e_uu = time()
                print(f"costing {e_uu - s_uu}s, saved uu_adj_mat...")
                sp.save_npz(self.data_path + f'/s_pre_uu_adj_mat.npz', UserUserNet)

                # build UserItemNet
                print("start generating UserUserNet")
                s_ui = time()
                users, items = [], []
                for user, item_seq in self.userIdx2sequence.items():
                    seq_len = len(item_seq)
                    users += [user] * seq_len
                    items += item_seq
                    UserItemNet = csr_matrix((np.ones(len(users)), (users, items)),
                                                shape=(self.user_num, self.item_num))

                e_ui = time()
                print(f"costing {e_ui - s_ui}s, saved ui_adj_mat...")
                sp.save_npz(self.data_path + f'/s_pre_ui_adj_mat.npz', UserItemNet)
                # build ItemItemNet


        print("1111111111111111")




        return self.Graph



# GraphDataCollector()