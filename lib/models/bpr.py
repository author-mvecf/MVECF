import torch.nn as nn
from torch.nn import LogSigmoid
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd
from lib.utils import save_pickle
import os
from tqdm import tqdm

__all__ = [
    "BPR",
    "BPRnov"
]


class MatrixFactorization(nn.Module):
    def __init__(self, n_items, n_users, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding_layer = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim)
        self.item_embedding_layer = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim)

    def forward(self, user, item):
        user = self.user_embedding_layer(user)
        item = self.item_embedding_layer(item)
        return (user * item).sum(axis=1)


def bpr_loss(x):
    return -LogSigmoid()(x[0] - x[1]).mean()


class BPR:
    def __init__(self, data, embedding_dim=30, lr=0.005, weight_decay=0.00001, device="cuda:0", max_iter=50000,
                 early_stop=True, tmp_save_path=None, tmp_load_path=None, batch_size=128):

        self.train_data = pd.DataFrame(data["train_data"].T, columns=['user_id', 'item_id', 'click'])
        self.train_indptr = data["train_indptr"]
        valid_data = data["valid_data"]
        self.n_users = data["n_users"]
        self.n_items = data["n_items"]

        self.epoch = 0

        self.data = data
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_iter = max_iter
        self.device = device
        self.not_clicked_data = []
        for user in range(self.n_users):
            indices = np.arange(self.train_indptr[user], self.train_indptr[user + 1])
            observed_items = self.train_data['item_id'][indices]
            nonobserved_items = list(set(range(self.n_items)) - set(observed_items))
            self.not_clicked_data.append(nonobserved_items)

        val_obs = pd.DataFrame(valid_data[:2].T[valid_data[2] == 1], columns=['user_id', 'positive_item_id'])
        val_non_obs = pd.DataFrame(valid_data[:2].T[valid_data[2] == 0], columns=['user_id', 'negative_item_id'])
        self.valid_pairs = pd.merge(val_obs, val_non_obs, on='user_id', how='inner')
        self.list_stop_cri = []
        self.is_early_stop = early_stop
        self.tmp_save_path = tmp_save_path
        self.tmp_load_path = tmp_load_path

        self.pu = None
        self.qi = None

        self.batch_size = batch_size
        self.train_loss = 0
        self.train_loss_rec = 0
        self.val_loss = 0
        self.val_loss_rec = 0
        self.map_valid = 0

        self.variable_list = [
            "pu", "qi", "n_users", "n_items", "embedding_dim", "lr",
            "weight_decay", "max_iter", "list_stop_cri", "val_loss", "map_valid"
        ]

    def fit(self):
        mf_model = MatrixFactorization(
            n_items=self.n_items, n_users=self.n_users, embedding_dim=self.embedding_dim,
        ).to(self.device)
        optimizer = torch.optim.Adam(mf_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.epoch = 0
        iterations = 0
        while True:
            d = self.sample_data()
            stop_flag = self.fit_iter(mf_model, optimizer, d, iterations)
            if iterations > self.max_iter or stop_flag:
                break
            iterations += 1

    def sample_data(self):
        sampled_data = self.train_data.sample(self.batch_size, replace=True)[['user_id', 'item_id']]
        negative_items = []
        for user in sampled_data["user_id"].values:
            negative_items.append(np.random.choice(self.not_clicked_data[user]))
        sample = {
            'user_ids': Variable(torch.FloatTensor(sampled_data['user_id'].values)).long().to(self.device),
            'positive_item_ids': Variable(torch.FloatTensor(sampled_data['item_id'].values)).long().to(self.device),
            'negative_item_ids': Variable(torch.FloatTensor(np.array(negative_items))).long().to(self.device),
        }
        return sample

    def fit_iter(self, mf_model, optimizer, d, iterations):
        stop_flag = False
        predict = [mf_model(item=d['positive_item_ids'], user=d['user_ids']),
                   mf_model(item=d['negative_item_ids'], user=d['user_ids'])]
        loss = bpr_loss(predict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iterations % 500 == 0:
            validation_score = self.validate(mf_model)
            self.map_valid = validation_score["map"]
            self.pu = mf_model.user_embedding_layer.weight.cpu().detach().numpy()
            self.qi = mf_model.item_embedding_layer.weight.cpu().detach().numpy()
            u_total = self.valid_pairs['user_id'].values
            i_total = self.valid_pairs['positive_item_id'].values
            j_total = self.valid_pairs['negative_item_id'].values
            pud = self.pu[u_total]
            qid = self.qi[i_total]
            qjd = self.qi[j_total]
            rui = np.sum(pud * qid, axis=1)
            ruj = np.sum(pud * qjd, axis=1)
            self.val_loss = -np.log(1 / (1 + np.exp(-rui + ruj))).sum()
            print(f'iteration: {iterations}, '
                  f'val recall@20: {validation_score["recall"]}, '
                  f'val precision@20: {validation_score["precision"]}, '
                  f'val map@20: {validation_score["map"]}, '
                  f'val loss: {self.val_loss}'
                  )
            self.list_stop_cri.append(self.val_loss)
            self.save_variables()
            self.epoch += 1
            if self.is_early_stop:
                if self.early_stop():
                    stop_flag = True

        return stop_flag

    def early_stop(self):
        if len(self.list_stop_cri) >= 30:
            if self.list_stop_cri[-30] <= min(self.list_stop_cri[-30:]) \
                    or np.all(max(abs(np.diff(self.list_stop_cri[-30:]))) < 0):
                return True
        return False

    def save_variables(self):
        if self.tmp_save_path is not None:
            save_path = os.path.join(self.tmp_save_path, "epoch_{}.pkl".format(self.epoch))

            variable_dict = {}
            for key in self.variable_list:
                variable_dict[key] = getattr(self, key)
            save_pickle(variable_dict, os.path.join(save_path))

    def validate(self, model, valid_type="valid", k=20):
        users = self.data["{}_data".format(valid_type)][0]
        items = self.data["{}_data".format(valid_type)][1]
        user_tensor = Variable(torch.FloatTensor(users)).long().to(self.device)
        item_tensor = Variable(torch.FloatTensor(items)).long().to(self.device)
        scores = model(item=item_tensor, user=user_tensor)
        y_score = scores.data.cpu().numpy()
        y_true = self.data["{}_data".format(valid_type)][2]
        indptr = self.data["{}_indptr".format(valid_type)]
        m = self.data["n_users"]
        n = self.data["n_items"]
        result_recall = 0
        result_precision = 0
        result_accuracy = 0
        result_map_at_k = 0
        for user in range(m):
            tmp_true = y_true[indptr[user]:indptr[user + 1]]
            total_num_relavant = tmp_true.sum()
            tmp_score = y_score[indptr[user]:indptr[user + 1]]
            sorted_index = tmp_score.argsort()[::-1]
            tmp_true = tmp_true[sorted_index]

            TN = (1 - tmp_true[k:]).sum()
            tmp_true = tmp_true[:k]
            # tmp_score = tmp_score[:k]
            # tmp_pred = np.ones(len(tmp_score))
            TP = sum(tmp_true)
            if total_num_relavant == 0:
                recall = 0
            else:
                recall = TP / min(total_num_relavant, k)
            precision = TP / k

            ap_at_k = 0
            for i in (np.where(tmp_true == 1)[0]):
                ap_at_k += tmp_true[:i + 1].sum() / min(total_num_relavant, i + 1) / k

            result_recall += recall / m
            result_precision += precision / m
            result_accuracy += (TP + TN) / n / m
            result_map_at_k += ap_at_k / m

        return dict(recall=result_recall, precision=result_precision, map=result_map_at_k)


class BPRnov(BPR):
    def __init__(self, data, distance, distance_threshold=0.9, novelty_rate=0.8, embedding_dim=30, lr=0.005,
                 weight_decay=0.00001, device="cuda:0", max_iter=50000, batch_size=128,
                 early_stop=True, tmp_save_path=None, tmp_load_path=None, ):
        super(BPRnov, self).__init__(data=data, embedding_dim=embedding_dim, lr=lr, weight_decay=weight_decay,
                                     device=device, max_iter=max_iter, early_stop=early_stop, batch_size=batch_size,
                                     tmp_save_path=tmp_save_path, tmp_load_path=tmp_load_path, )
        self.distance = distance
        self.distance_threshold = distance_threshold
        self.novelty_rate = novelty_rate

        self.valid_pairs['distance'] = self.distance[
            self.valid_pairs['positive_item_id'], self.valid_pairs['negative_item_id']
        ]
        self.valid_pairs = self.valid_pairs[self.valid_pairs['distance'] < distance_threshold]
        self.variable_list = [
            "pu", "qi", "n_users", "n_items", "distance_threshold", "novelty_rate", "embedding_dim", "lr",
            "weight_decay", "max_iter", "list_stop_cri", "val_loss", "map_valid"
        ]

    def sample_data(self):
        sampled_data = self.train_data.sample(self.batch_size, replace=True)[['user_id', 'item_id']]
        negative_items = []
        for idx, row in sampled_data.iterrows():
            user = row["user_id"]
            item = row["item_id"]
            user_not_clicked_data = np.array(self.not_clicked_data[user])
            if np.random.binomial(1, self.novelty_rate, 1)[0]:
                distance = self.distance[item, user_not_clicked_data]
                if distance.min() < self.distance_threshold:
                    user_not_clicked_data = user_not_clicked_data[distance < self.distance_threshold]
            neg_item = np.random.choice(user_not_clicked_data)
            negative_items.append(neg_item)
        sample = {
            'user_ids': Variable(torch.FloatTensor(sampled_data['user_id'].values)).long().to(self.device),
            'positive_item_ids': Variable(torch.FloatTensor(sampled_data['item_id'].values)).long().to(self.device),
            'negative_item_ids': Variable(torch.FloatTensor(np.array(negative_items))).long().to(self.device),
        }
        return sample
