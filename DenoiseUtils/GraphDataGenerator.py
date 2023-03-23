import torch
from recbole.model.abstract_recommender import SequentialRecommender
import scipy.sparse as sp
from time import time
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


class GraphDataCollector(SequentialRecommender):
    def __init__(self, config, dataset, train_data):
        super().__init__(config, dataset)
        self.Graph = None
        self.data_path = config['data_path']
        self.user_num = dataset.num(self.USER_ID)
        self.item_num = dataset.num(self.ITEM_ID)
        self.behavior_num = dataset.num(self.BEHAVIOR_ID)
        self.item2Idx = dataset.field2token_id['item_id']
        self.user2Idx = dataset.field2token_id['item_id']
        self.userSet = train_data.dataset.inter_feat.interaction['user_id']
        self.item_seq = train_data.dataset.inter_feat.interaction['item_id_list']
        self.behavior_seq = train_data.dataset.inter_feat.interaction['behavior_type_list']

        self.tempUserSet = self.userSet.numpy().tolist()
        self.tempBehaviorSeq = self.behavior_seq.numpy().tolist()
        self.userIdx2itemSeq = dict(zip(self.userSet, self.item_seq))
        self.userId2behaviorSeq = dict(zip(self.tempUserSet, self.tempBehaviorSeq))

        # self.userItemSet = dict(zip(self.userSet, set(self.item_id_list)))
        self.itemSeq2user = {}
        self.user_dist = [0 for _ in range(self.user_num)]
        self.item_dist = [0 for _ in range(self.item_num)]
        # self.getSparseGraph()

    def getSparseGraph(self):
        UserUserNet, UserItemNet, ItemItemNet = None, None, None
        print("loading pre adjacency matrix")
        if self.Graph is None:
            try:
                # path = '../dataset/tmall/s_pre_adj_mat.npz'
                adj_mat = sp.load_npz(f'{self.data_path}/ss_pre_adj_mat.npz')
                print("successfully loaded pre_adj_mat...")
                norm_adj = adj_mat
            except:
                # build ItemItemNet
                # print("start generating ItemItemNet")
                # s_ii = time()
                # c = 0
                # ii_adj_mat = torch.zeros(self.item_num, self.item_num, dtype=torch.int64)
                # for user, item_seq in self.userIdx2itemSeq.items():
                #     c += 1
                #     if c % 1000 == 0:
                #         print(f'{c} users load, {time()-s_ii}s cost')
                #     behavior_seq = torch.Tensor(self.userId2behaviorSeq[user.item()])
                #     mask = torch.eq(item_seq, 0).logical_not_()
                #     item_seq = torch.masked_select(item_seq, mask)
                #     behavior_seq = torch.masked_select(behavior_seq, mask)
                #
                #     # purchase_indices = [p for p, b in enumerate(behavior_seq) if b == 2]
                #     last_purchase = None
                #     for i, behavior in enumerate(behavior_seq):
                #         if behavior == 2:  # purchase
                #             if last_purchase is None:
                #                 last_purchase = i
                #             else:
                #                 for j in range(last_purchase + 1, i):
                #                     if behavior_seq[j] == 1:  # view
                #                         source_item = item_seq[j]
                #                         target_item = item_seq[i]
                #                         ii_adj_mat[target_item][source_item] = 1
                #                         ii_adj_mat[source_item][target_item] = 1
                #                 last_purchase = i
                #         elif behavior == 1:
                #             source_item = item_seq[i]
                #             if i > 0 and behavior_seq[i-1] == 1:
                #                 target_item = item_seq[i-1]
                #                 ii_adj_mat[source_item][target_item] = 1
                #             if i < len(behavior_seq) - 1 and behavior_seq[i+1] == 1:
                #                 target_item = item_seq[i+1]
                #                 ii_adj_mat[source_item][target_item] = 1
                # idx = torch.nonzero(ii_adj_mat).T
                # data = ii_adj_mat[idx[0], idx[1]]
                # rows = idx[0]
                # cols = idx[1]
                # ItemItemNet = sp.csr_matrix((data, (rows, cols)), ii_adj_mat.shape)
                # e_ii = time()
                # print(f"costing {e_ii - s_ii}s, saved ii_adj_mat...")
                # sp.save_npz(f'{self.data_path}/ss_pre_ii_adj_mat.npz', ItemItemNet)


                # build UserUserNet
                # print("start generating UserUserNet")
                # rating_mat = torch.zeros((self.user_num, self.item_num))
                # for user, item_seq in self.userIdx2itemSeq.items():
                #     rating_mat[user][item_seq] = 1
                #
                # s_uu = time()
                # rows, cols, weights = [], [], []
                # c = 0
                # for user1, item_seq1 in self.userIdx2itemSeq.items():
                #     c += 1
                #     if c % 1000 == 0:
                #         print(f'{c} users load, {time()-s_uu}s cost')
                #     for user2, item_seq2 in self.userIdx2itemSeq.items():
                #         if user2 < user1 or user2 == user1:
                #             continue
                #         else:
                #             neighbors_num = torch.sum(rating_mat[user1] * rating_mat[user2])
                #             if neighbors_num == 0:
                #                 continue
                #             weight = neighbors_num / (torch.sum(rating_mat[user1]) + torch.sum(rating_mat[user2]) - neighbors_num)
                #             # print("=======================================================neighbors================================================")
                #             rows.append(user1)
                #             cols.append(user2)
                #             weights.append(weight)
                #             rows.append(user2)
                #             cols.append(user1)
                #             weights.append(weight)
                #
                # UserUserNet = csr_matrix((weights, (rows, cols)), shape=(self.user_num, self.user_num))
                # e_uu = time()
                # print(f"costing {e_uu - s_uu}s, saved uu_adj_mat...")
                # sp.save_npz(f'{self.data_path}/ss_pre_uu_adj_mat.npz', UserUserNet)

                # build UserItemNet
                # print("start generating UserItemNet")
                # s_ui = time()
                # users, items = [], []
                # c = 0
                # for user, item_seq in self.userIdx2itemSeq.items():
                #     c += 1
                #     if c % 1000 == 0:
                #         print(f'{c} users load, {time()-s_ui}s cost')
                #     mask = torch.eq(item_seq, 0).logical_not_()
                #     new_item_seq = torch.masked_select(item_seq, mask)
                #     seq_len = len(new_item_seq)
                #     users += [user] * seq_len
                #     items += new_item_seq
                # UserItemNet = csr_matrix((np.ones(len(users)), (users, items)),
                #                              shape=(self.user_num, self.item_num))
                #
                # e_ui = time()
                # print(f"costing {e_ui - s_ui}s, saved ui_adj_mat...")
                # sp.save_npz(f'{self.data_path}/ss_pre_ui_adj_mat.npz', UserItemNet)

                print("generating full adjacency matrix")
                ItemItemNet = sp.load_npz(f'{self.data_path}/ss_pre_ii_adj_mat.npz')
                UserUSerNet = sp.load_npz(f'{self.data_path}/ss_pre_uu_adj_mat.npz')
                UserItemNet = sp.load_npz(f'{self.data_path}/ss_pre_ui_adj_mat.npz')
                s = time()
                adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                UI = UserItemNet.tolil()
                II = ItemItemNet.tolil()
                UU = UserUSerNet.tolil()

                # adj_mat[:self.user_num, :self.user_num] = UU
                adj_mat[:self.user_num, self.user_num:] = UI
                # adj_mat[self.user_num:, :self.user_num] = UI.T
                adj_mat[self.user_num:, self.user_num:] = II
                # adj_mat.todok()

                array_adj_mat = adj_mat.toarray()
                rowsum = np.expand_dims(np.count_nonzero(array_adj_mat, axis=1), 0)
                d_inv = np.power(rowsum, -0.5).flatten()  # Mu, Mj
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)  # 对角线矩阵
                adj_mat[:self.user_num, :self.user_num] = UU
                adj_mat.todok()
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                print(f"costing {time() - s}s, saved adj_mat...")
                sp.save_npz(f'{self.data_path}/ss_pre_adj_mat.npz', norm_adj)

            self.Graph = self._convert_sp_mat_to_torch_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device).double()

        return self.Graph

    def _convert_sp_mat_to_torch_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))