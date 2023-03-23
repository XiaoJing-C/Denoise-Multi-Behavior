import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from torch.nn.init import xavier_uniform_, xavier_normal_
from DenoiseUtils.utils import get_model
from DenoiseUtils.GraphDataGenerator import GraphDataCollector


class Denoise(SequentialRecommender):

    def __init__(self, config, dataset):
        super(Denoise, self).__init__(config, dataset)

        self.hidden_size = config['hidden_size']
        self.user_num = dataset.num(self.USER_ID)
        self.item_num = dataset.num(self.ITEM_ID)
        self.behavior_num = dataset.num(self.BEHAVIOR_ID)
        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size, padding_idx=0)
        # self.item_embedding = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.behavior_embedding = nn.Embedding(self.behavior_num, self.hidden_size, padding_idx=0)

        self.emb_drop_out = config['emb_drop_out']
        self.att_drop_out = config['att_drop_out']

        self.tau = 100
        self.filter_drop_rate = 0.0
        self.layer_num = config['num_layer']
        self.LayerNorm = nn.BatchNorm1d(self.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(self.hidden_size, 1e-12)
        self.prev_next_weight = nn.Parameter(torch.randn(2, 1))
        self.emb_dropout = nn.Dropout(self.emb_drop_out)
        self.att_dropout = nn.Dropout(self.att_drop_out)

        self.conv_item = nn.Conv2d(self.item_num, self.item_num, (1, 2))
        self.conv_user = nn.Conv2d(self.user_num, self.user_num, (1, 2))
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.loss_fuc = nn.CrossEntropyLoss()

        self.soft_att_out = SoftAttnout(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            session_len=self.max_seq_length,
            batch_norm=True,
            feat_drop=self.att_drop_out,
            activation=nn.PReLU(self.hidden_size),
        )

        # self.graph = GraphDataCollector.getSparseGraph()
        self.apply(self._init_weights)

        # 初始化sub_model
        self.sub_model = get_model(config['sub_model'])(config, dataset).to(config['device'])
        self.sub_model_name = config['sub_model']
        self.item_embedding = self.sub_model.item_embedding

    def long_term_consistency(self, item_seq_emb, long_term_representation, mask):
        long_term_score = self.soft_att_out(item_seq_emb, long_term_representation, mask)
        return long_term_score

    # def short_term_consistency(self, prev_next_purchase, item_seq_emb, item_seq, behavior_seq):
    #
    #     batch_size = len(item_seq)
    #     seq_len = self.max_seq_length
    #     # prev_next_score = []
    #     prev_score = [[] for _ in range(batch_size)]
    #     next_score = [[] for _ in range(batch_size)]
    #     # purchase_mask = [[] for _ in range(batch_size)]
    #     purchase_mask = (behavior_seq == 2).float()
    #     for b in range(batch_size):
    #         iter_item_seq = item_seq[b]
    #         iter_behavior_seq = behavior_seq[b]
    #         # purchase_mask[b] = (iter_behavior_seq == 2).float()
    #         # view_items = [iter_item_seq[i] for i in range(seq_len) if iter_behavior_seq[i] == 1]
    #
    #         # for i in iter_item_seq:
    #         one = torch.tensor(1, device=self.device)
    #         zero = torch.tensor(0, device=self.device)
    #         prev_purchase_items = [prev_next_purchase[b].get(int(i))[0] for i in iter_item_seq]
    #         next_purchase_items = [prev_next_purchase[b].get(int(i))[1] for i in iter_item_seq]
    #         for i in range(seq_len):
    #             if iter_behavior_seq[i] == 2:
    #                 next_score[b].append(one)
    #                 prev_score[b].append(one)
    #             elif iter_behavior_seq[i] == 1:
    #                 if prev_purchase_items[i] is not None:
    #                     i1, i2 = iter_item_seq[i], prev_purchase_items[i]
    #                     i1_emb, i2_emb = item_seq_emb[i1].reshape(1, self.hidden_size), item_seq_emb[i2].reshape(1, self.hidden_size)
    #                     sim = (F.cosine_similarity(i1_emb, i2_emb) + 1) / 2.0
    #                 else:
    #                     sim = zero
    #                 prev_score[b].append(sim)
    #                 if next_purchase_items[i] is not None:
    #                     i1, i2 = iter_item_seq[i], next_purchase_items[i]
    #                     i1_emb, i2_emb = item_seq_emb[i1].reshape(1, self.hidden_size), item_seq_emb[i2].reshape(1, self.hidden_size)
    #                     sim = (F.cosine_similarity(i1_emb, i2_emb) + 1) / 2.0
    #                 else:
    #                     sim = zero
    #                 next_score[b].append(sim)
    #             else:
    #                 next_score[b].append(-1)
    #                 prev_score[b].append(-1)
    #             # prev_next_score.append(torch.cat([prev_score[b][i], next_score[b][i]]))
    #
    #     prev_score, next_score = torch.Tensor(prev_score), torch.Tensor(next_score)
    #     prev_next_score = torch.stack([next_score, prev_score], dim=2)
    #     # prev_next_score = torch.stack([prev_score, next_score], dim=2)
    #     # prev_next_weight = torch.tensor([0.7, 0.3]).to(self.device)
    #     # short_term_score =  self.sigmoid(torch.matmul(prev_next_score.to(self.device), self.prev_next_weight))
    #     # short_term_score =  torch.matmul(prev_next_score.to(self.device), self.prev_next_weight)
    #     # short_term_score = torch.mean(prev_next_score, dim=2)
    #
    #     short_term_score = prev_next_score
    #     # short_term_score = self.sigmoid(prev_next_score)
    #
    #     return short_term_score, purchase_mask
    def short_term_consistency(self, prev_next_purchase, item_seq_emb, item_seq, behavior_seq):
        batch_size, seq_len = behavior_seq.size()
        prev_score = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)
        next_score = torch.zeros(batch_size, seq_len, dtype=torch.float32, device=self.device)
        purchase_mask = (behavior_seq == 2).float()
        one = torch.tensor(1.0, device=self.device)

        prev_purchase_items = torch.zeros_like(item_seq, dtype=torch.long)
        next_purchase_items = torch.zeros_like(item_seq, dtype=torch.long)

        for b in range(batch_size):
            prev_purchase_items[b] = torch.Tensor([prev_next_purchase[b][i][0] for i in range(seq_len)])
            next_purchase_items[b] = torch.Tensor([prev_next_purchase[b][i][1] for i in range(seq_len)])

        purchase_indices = torch.nonzero(behavior_seq == 2)
        next_score[purchase_indices[:,0], purchase_indices[:,1]] = one
        prev_score[purchase_indices[:,0], purchase_indices[:,1]] = one
        view_indices = torch.nonzero(behavior_seq == 1)
        if len(view_indices) > 0:
            i1_emb = item_seq_emb[item_seq[view_indices[:, 0], view_indices[:, 1]]].unsqueeze(1)
            i2_emb = item_seq_emb[prev_purchase_items[view_indices[:, 0], view_indices[:, 1]]].unsqueeze(1)
            sim = (F.cosine_similarity(i1_emb, i2_emb, dim=2) + 1) / 2.0
            prev_score[view_indices[:, 0], view_indices[:, 1]] = sim.squeeze(-1)
            i2_emb = item_seq_emb[next_purchase_items[view_indices[:, 0], view_indices[:, 1]]].unsqueeze(1)
            sim = (F.cosine_similarity(i1_emb, i2_emb, dim=2) + 1) / 2.0
            next_score[view_indices[:, 0], view_indices[:, 1]] = sim.squeeze(-1)
        prev_next_score = torch.stack([next_score, prev_score], dim=2)
        short_term_score = prev_next_score

        return short_term_score, purchase_mask

    def loss_filter(self, user, item_seq, item_seq_len, item_emb, user_emb, interaction, mask, train_flag):
        # item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = item_emb[item_seq]
        item_seq_emb = (item_seq_emb * mask).unsqueeze(-2)  # [B, L, 1, H]
        # user_emb = self.user_embedding(user).unsqueeze(-2).unsqueeze(-1)  # [B, 1, H, 1]
        user_emb = user_emb[user]
        user_emb = user_emb.unsqueeze(-2).unsqueeze(-1)
        # item_seq_emb = self.item_embedding(item_seq)  # [B, L, 1, H]
        # item_seq_emb = (item_seq_emb * mask).unsqueeze(-2)
        # user_emb = self.user_embedding(user).unsqueeze(-2).unsqueeze(-1)

        pos_score = torch.matmul(item_seq_emb, user_emb)  # [B, L, 1, 1]

        filter_drop_rate = self.filter_drop_rate

        if train_flag:
            neg_items = interaction[self.NEG_ITEM_ID]
            neg_items = neg_items.unsqueeze(-1).expand(item_seq.shape)
            neg_items_emb = item_emb[neg_items]  # [B, L, H]
            # neg_items_emb = self.item_embedding(neg_items)
            neg_items_emb = (neg_items_emb * mask).unsqueeze(-2)  # [B, L, 1, H]
            neg_score = torch.matmul(neg_items_emb, user_emb)  # [B, L, 1, H]
        else:
            neg_score = torch.zeros_like(pos_score)
            filter_drop_rate = 0.2

        loss = -torch.log(1e-10 + torch.sigmoid(pos_score - neg_score)).squeeze(-1).squeeze(-1)  # [B, L]
        loss = loss * mask.squeeze(-1)
        loss_sorted, ind_sorted = torch.sort(loss, descending=False, dim=-1)
        num_remember = (filter_drop_rate * item_seq_len).squeeze(-1).long()

        loss_filter_flag = torch.zeros_like(item_seq)

        for index, filtered_item_num in enumerate(num_remember):
            loss_index = ind_sorted[index][-filtered_item_num:]
            loss_filter_flag[index][loss_index] = 1
            if filter_drop_rate != 0:
                loss[index][loss_index] *= 0
        loss_filter_flag = loss_filter_flag * mask.squeeze(-1)
        return loss, loss_filter_flag

    def method_name(self, generated_seq, generated_seq_emb):
        row_indexes, col_id = torch.where(generated_seq.gt(0))
        row_flag = row_indexes[0]
        index_flag = -1
        col_index = []
        for row_index in row_indexes:
            if row_index == row_flag:
                index_flag += 1
                col_index.append(index_flag)
            else:
                index_flag = 0
                col_index.append(index_flag)
                row_flag = row_index
        col_index = torch.tensor(col_index, device=self.device)
        denoising_seq = torch.zeros_like(generated_seq)
        denoising_seq_emb = torch.zeros_like(generated_seq_emb)
        denoising_seq[row_indexes, col_index] = generated_seq[row_indexes, col_id]
        denoising_seq_emb[row_indexes, col_index, :] = generated_seq_emb[row_indexes, col_id, :]
        return denoising_seq, denoising_seq_emb

    def generate_pos_seq(self, item_seq, item_seq_len, item_emb, long_term_score, short_term_score, mask, purchase_mask):
        # item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = item_emb[item_seq]
        mask = mask.squeeze()

        # Todo 可调整
        long_term_gumbel_softmax_rst = F.gumbel_softmax(long_term_score, tau=self.tau, hard=True)
        short_term_gumbel_softmax_rst = F.gumbel_softmax(short_term_score, tau=self.tau, hard=True)
        # x = torch.count_nonzero(short_term_gumbel_softmax_rst, dim=1).reshape(-1, 1)
        # x1 = long_term_gumbel_softmax_rst[:, :, 1]
        long_term_denoising_seq_flag = long_term_gumbel_softmax_rst[:, :, 1] * mask
        short_term_denoising_seq_flag = short_term_gumbel_softmax_rst[:, :, 1].to(self.device) * mask
        # short_term_denoising_seq_flag = short_term_gumbel_softmax_rst.squeeze(-1) * mask
        # short_term_denoising_seq_flag = short_term_gumbel_softmax_rst.to(self.device) * mask
        # mark = torch.rand_like(purchase_mask) < 0.2
        # purchase_mask[mark] = 0

        # num_ones = torch.sum(purchase_mask, dim=1)
        # num_zeros = torch.floor(0.1 * num_ones).int()
        # rand_tensor = torch.rand(purchase_mask.shape)
        # ones_tensor = torch.ones(purchase_mask.shape)
        # zeros_mask = (rand_tensor < num_zeros.float() / num_ones.float()).float()
        # purchase_mask = purchase_mask * (ones_tensor - zeros_mask)
        # short_term_denoising_seq_flag = short_term_denoising_seq_flag * (1 - purchase_mask)
        # noisy_flag = long_term_denoising_seq_flag
        # noisy_flag = short_term_denoising_seq_flag
        noisy_flag = long_term_denoising_seq_flag * short_term_denoising_seq_flag

        pos_flag = (1 - noisy_flag) * mask

        pos_seq_emb = item_seq_emb * pos_flag.unsqueeze(-1)

        pos_seq = item_seq * pos_flag
        pos_seq[pos_seq != pos_seq] = 0

        pos_seq_len = torch.sum(pos_flag, dim=-1)

        # 如果序列为0 stamp会报错，因此这里将序列长度为0 的保留第一个
        pos_seq_len[pos_seq_len.eq(0)] = 1

        # TODO: [1,2,3,4] -> [1,0,3,4] or [1,3,4]
        clean_seq_percent = torch.sum(pos_seq_len, dim=0) / item_seq_len.sum() * 100
        denoising_seq, denoising_seq_emb = self.method_name(pos_seq, pos_seq_emb)
        pos_seq = denoising_seq
        pos_seq_emb = denoising_seq_emb
        neg_seq_len = torch.squeeze(item_seq_len)

        return pos_seq, pos_seq_emb, pos_seq_len.long(), item_seq, neg_seq_len, clean_seq_percent

    def find_prev_next_purchase1(self, item_seq, behavior_seq):
        prev_next_purchase = []
        for b in range(len(item_seq)):
            iter_item_seq = item_seq[b]
            iter_behavior_seq = behavior_seq[b]
            seq_len = self.max_seq_length
            prev_purchase = {}
            next_purchase = {}
            last_purchase = 0
            temp = []
            for i in range(seq_len):
                item = int(iter_item_seq[i])
                behavior = iter_behavior_seq[i]
                if behavior == 2:
                    last_purchase = item
                if behavior == 1:
                    if last_purchase == 0:
                        prev_purchase[item] = 0
                    else:
                        prev_purchase[item] = last_purchase

                    for j in range(i+1, seq_len):
                        if iter_behavior_seq[j] == 2:
                            next_purchase[item] = int(iter_item_seq[j])
                            # t = next_purchase.get(item)
                            break
                    else:
                        next_purchase[item] = 0
                if behavior == 0 or behavior == 2:
                    prev_purchase[item], next_purchase[item] = 0, 0
                temp.append([prev_purchase.get(item), next_purchase.get(item)])
            prev_next_purchase.append(temp)

        return prev_next_purchase

    def find_prev_next_purchase(self, item_seq, behavior_seq):

        batch_size, seq_len = behavior_seq.size()
        prev_next_purchase = torch.zeros((batch_size, seq_len, 2), dtype=torch.long, device=self.device)
        prev_purchase = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        next_purchase = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        last_purchase = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(seq_len):
            item = item_seq[:, i]
            behavior = behavior_seq[:, i]
            purchase_mask = (behavior == 2)

            prev_purchase[:, i] = torch.where(last_purchase != 0, last_purchase, prev_purchase[:, i])
            last_purchase = torch.where(purchase_mask, item, last_purchase)
            sub_item_seq = item_seq[:, i + 1:]
            sub_behavior_seq = behavior_seq[:, i + 1:]
            next_purchase_mask = (sub_behavior_seq == 2) * sub_behavior_seq
            if next_purchase_mask.size(1) != 0:
                next_purchase_indices = next_purchase_mask.max(dim=1).indices
                next_purchase[:, i] = sub_item_seq[torch.arange(len(next_purchase_indices)), next_purchase_indices]

            prev_next_purchase[:, i, 0] = prev_purchase[:, i]
            prev_next_purchase[:, i, 1] = next_purchase[:, i]

        return prev_next_purchase

    def forward(self, interaction, train_flag=True):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN].unsqueeze(1)
        behavior_seq = interaction[self.BEHAVIOR_SEQ]
        user = interaction[self.USER_ID]

        # item_seq_emb = self.item_embedding(item_seq)
        # behavior_seq_emb = self.behavior_embedding(behavior_seq)
        # user_emb = self.user_embedding(user)

        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        # target_embedding = self.item_embedding(target_item)

        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)

        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        adj_mat = self.graph
        adj_mat = adj_mat.float()
        for layer in range(self.layer_num):
            all_emb = torch.sparse.mm(adj_mat, all_emb)

        p_user_emb, p_item_emb = torch.split(all_emb, [self.user_num, self.item_num])
        p_user_emb, p_item_emb = self.LayerNorm1(p_user_emb), self.LayerNorm1(p_item_emb)
        # concat_shuffled_and_origin_item = torch.stack((p_item_emb, items_emb), dim=-1)  # [B len 2xh]
        # concat_shuffled_and_origin_item = self.conv_item(concat_shuffled_and_origin_item)  # [B len h 1]
        # concat_shuffled_and_origin_item = torch.squeeze(concat_shuffled_and_origin_item)  # [B len h]
        # concat_shuffled_and_origin_item = self.emb_dropout(concat_shuffled_and_origin_item)  # [B len h]
        # concat_shuffled_and_origin_item = nn.ReLU(inplace=True)(concat_shuffled_and_origin_item)  # [B len h]
        #
        # concat_shuffled_and_origin_user = torch.stack((p_user_emb, users_emb), dim=-1)  # [B len 2xh]
        # concat_shuffled_and_origin_user = self.conv_user(concat_shuffled_and_origin_user)  # [B len h 1]
        # concat_shuffled_and_origin_user = torch.squeeze(concat_shuffled_and_origin_user)  # [B len h]
        # concat_shuffled_and_origin_user = self.emb_dropout(concat_shuffled_and_origin_user)  # [B len h]
        # concat_shuffled_and_origin_user = nn.ReLU(inplace=True)(concat_shuffled_and_origin_user)  # [B len h]
        # # size = [batchSize, user_num_per_batch, 1]
        #
        # p_user_emb, p_item_emb = concat_shuffled_and_origin_user, concat_shuffled_and_origin_item
        mask = mask.unsqueeze(2)

        long_term_representation = p_user_emb

        long_term_score = self.long_term_consistency(
            item_seq_emb=p_item_emb[item_seq],
            long_term_representation=long_term_representation[user],
            mask=mask)

        prev_next_purchase = self.find_prev_next_purchase(item_seq, behavior_seq)
        short_term_score, purchase_mask = self.short_term_consistency(prev_next_purchase, p_item_emb, item_seq, behavior_seq)
        loss, loss_filter_flag = self.loss_filter(user, item_seq, item_seq_len, p_item_emb, p_user_emb, interaction, mask, train_flag)

        pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, clean_seq_percent = self.generate_pos_seq(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            item_emb=p_item_emb,
            long_term_score=long_term_score,
            short_term_score=short_term_score,
            mask=mask,
            purchase_mask=purchase_mask
        )

        return pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, p_item_emb, p_user_emb, clean_seq_percent, loss.sum(-1).mean()


    # def propagate_embeds(self):
    #     adj_mat = self.graph

    def calculate_loss_denoise(self, interaction, graph, drop_rate, tau):

        self.tau = tau
        self.filter_drop_rate = 0.2 - drop_rate
        self.graph = graph

        pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, item_emb, user_emb, clean_seq_percent, loss_filter_loss = self.forward(interaction)

        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        all_items_emb = self.item_embedding.weight[:self.item_num]
        # using the denoisied embedding calculate predict loss
        sub_model_output = self.sub_model_forward(pos_seq, pos_seq_emb, pos_seq_len, user, user_emb)
        seq_representation, delete_index = self.denoising_seq_gather(pos_seq, sub_model_output)
        # scores = torch.matmul(seq_representation, item_emb.transpose(0,1))
        scores = torch.matmul(seq_representation, all_items_emb.transpose(0,1))

        target_item = interaction[self.ITEM_ID].unsqueeze(1)
        target_item = target_item.squeeze()

        generated_seq_loss = self.loss_fuc(scores, target_item)
        total_loss = loss_filter_loss + generated_seq_loss

        return total_loss, generated_seq_loss, loss_filter_loss, clean_seq_percent

    # def cal_curriculum_batch_id(self, drop_rate, element_wise_reconstruction_loss):
    #     loss_sorted, ind_sorted = torch.sort(element_wise_reconstruction_loss, descending=False)
    #     remember_rate = 1 - drop_rate
    #     num_remember = int(remember_rate * len(loss_sorted))
    #     ind_update = ind_sorted[:num_remember]
    #     return ind_update

    def denoising_seq_gather(self, generated_seq, seq_output):
        generated_item_seq_len = torch.sum(generated_seq.gt(0), 1)

        # 算出长度为0的则是全为噪声项的序列，这里先记录其index
        delete_index = torch.where(generated_item_seq_len.eq(0))[0]
        # 将index减一 ，防止数组越上界
        generated_item_seq_len = generated_item_seq_len - 1
        # 将index为-1 的项置为0，防止数组越下界
        generated_item_seq_len = generated_item_seq_len * generated_item_seq_len.gt(0)
        if self.sub_model_name in ['SRGNN', 'GCSAN', 'Caser', 'NARM', 'DSAN', 'STAMP']:
            seq_output = seq_output
        elif self.sub_model_name == 'fmlp':
            seq_output = seq_output[:, -1, :]  # delete masked token
        else:
            seq_output = self.gather_indexes(seq_output, generated_item_seq_len)  # [B H]
        return seq_output, delete_index

    def sub_model_forward(self, generated_seq, pos_seq_emb, denoising_seq_len, user, user_emb):
        if self.sub_model_name == 'BERT4Rec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'GRU4Rec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'SASRec':
            seq_output = self.sub_model.forward_denoising(generated_seq, pos_seq_emb)
        elif self.sub_model_name == 'Caser':
            seq_output = self.sub_model.forward_denoising(user, generated_seq, pos_seq_emb, user_emb)
        elif self.sub_model_name == 'NARM':
            seq_output = self.sub_model.forward_denoising(generated_seq, denoising_seq_len, pos_seq_emb)
        elif self.sub_model_name == 'DSAN':
            seq_output, _ = self.sub_model.forward(generated_seq)
        elif self.sub_model_name == 'fmlp':
            seq_output = self.sub_model.forward(generated_seq)
        elif self.sub_model_name == 'STAMP':
            seq_output = self.sub_model.forward_denoising(generated_seq, denoising_seq_len, pos_seq_emb)
        else:
            raise ValueError(f'Sub_model [{self.sub_model_name}] not support.')
        return seq_output

    def predict(self, interaction):
        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        # pos_seq, pos_seq_emb, pos_seq_len, neg_seq, neg_seq_len, item_emb, user_emb, clean_seq_percent, loss_filter_loss
        pos_seq, generated_seq, denoising_seq_len, temp1_, temp2_, embedding_level_denoising_emb, user_emb, clean_seq_percent, loss = self.forward(
            interaction,
            train_flag=False)
        seq_output = self.sub_model_forward(generated_seq, denoising_seq_len, user)

        seq_output, _ = self.denoising_seq_gather(generated_seq, seq_output)

        test_item = interaction[self.ITEM_ID]
        # test_item_emb = self.item_embedding(test_item)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID] if self.sub_model_name == 'Caser' else None
        pos_seq, pos_seq_emb, denoising_seq_len, _, _, embedding_level_denoising_emb, p_user_emb, pre, loss = self.forward(
            interaction,
            train_flag=False)
        seq_output = self.sub_model_forward(pos_seq, pos_seq_emb, denoising_seq_len, user, p_user_emb)

        seq_output, _ = self.denoising_seq_gather(pos_seq, seq_output)

        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        return scores, pre

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)


class SoftAttnout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            session_len,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(session_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_w2 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_w3 = nn.Linear(hidden_dim, 2, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.mlp_n_ls = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat, long_term_representation, mask):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = feat * mask
        feat = self.feat_drop(feat)
        feat = feat * mask
        feat_i = self.fc_w1(feat)
        feat_i = feat_i * mask
        feat_u = self.fc_w2(long_term_representation.unsqueeze(1))  # (batch_size * embedding_size)

        long = self.fc_w3(F.tanh(feat_i + feat_u)) * mask
        alph_long = self.sigmoid(long)

        alph_long_score = alph_long.squeeze()
        return alph_long_score
