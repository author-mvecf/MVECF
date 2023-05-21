# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp

from lib.models.lightGCN.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)


class ImplicitCF(object):
    """Data processing class for GCN models which use implicit feedback.

    Initialize train and test set, create normalized adjacency matrix and sample data for training epochs.

    """

    def __init__(
            self,
            train,
            test=None,
            adj_dir=None,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
            seed=None,
            alpha=10,
            factor_params=None,
            reg_param_mv=1,
            gamma=3,
            positive_score_cri=None,
            num_pos_to_neg=None,
            num_neg_to_pos=None,
    ):
        """Constructor

        Args:
            adj_dir (str): Directory to save / load adjacency matrices. If it is None, adjacency
                matrices will be created and will not be saved.
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                test can be None, if so, we only process the training data.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            seed (int): Seed.

        """
        self.user_idx = None
        self.item_idx = None
        self.adj_dir = adj_dir
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.train, self.test = self._data_processing(train, test)
        self.interact_status = None
        self._init_train_data()

        self.alpha = alpha
        self.dense_rating = None
        self.dense_weight = None
        self.avg_true_beta = None
        self.num_holding = None
        self.factor_params = factor_params
        self.reg_param_mv = reg_param_mv
        self.gamma = gamma
        self.positive_score_cri = positive_score_cri
        self.num_pos_to_neg = num_pos_to_neg
        self.num_neg_to_pos = num_neg_to_pos
        if factor_params is not None:
            self.prepare_mv_params()
        random.seed(seed)

    def prepare_mv_params(self):
        sig2_M = self.factor_params["factor_variance"]
        beta = self.factor_params["beta"]
        sig2_eps = self.factor_params["sig_eps_square"]
        Sigma = np.matmul(np.matmul(beta, sig2_M), beta.T) + np.diag(sig2_eps)
        diag_Sigma = np.diag(Sigma)
        mu_items = self.cal_item_means()
        # generate dense rating and weight
        interact = self.interact_status
        new_interact = {self.col_user: [], self.col_item + "_interacted": []}
        self.dense_rating = []
        self.dense_weight = []
        self.avg_true_beta = []
        self.num_holding = []
        interaction_data = interact.iterrows()
        count_zero_positive = 0
        for _, user_interaction_data in interaction_data:
            user = user_interaction_data[self.col_user]
            train_items = np.array(list(user_interaction_data[self.col_item + "_interacted"]))
            train_items.sort()
            train_ratings = np.ones(len(train_items))
            self.num_holding.append(len(train_items))
            self.avg_true_beta.append(beta[train_items].mean(axis=0))
            i_user, r_user, w_user, new_positive_items = self._get_total_items_ratings_init(
                user, train_items, train_ratings, mu_items, diag_Sigma)
            if not new_positive_items:
                count_zero_positive += 0
                continue
            new_interact[self.col_user].append(user)
            new_interact[self.col_item + "_interacted"].append(new_positive_items)
            self.dense_rating.append(r_user)
            self.dense_weight.append(w_user)
        print("zero_positive: ", count_zero_positive)

        self.num_holding = np.array(self.num_holding)
        self.avg_true_beta = np.array(self.avg_true_beta)

        self.dense_rating = np.array(self.dense_rating)
        self.dense_rating = self.dense_rating - self.dense_rating.mean()
        self.dense_weight = np.array(self.dense_weight)
        self.interact_status = pd.DataFrame.from_dict(new_interact)

    def cal_item_means(self):
        return np.matmul(self.factor_params["beta"], self.factor_params["factor_mean"])

    def _get_total_items_ratings_init(self, user, train_items, train_ratings, mu_items, diag_Sigma):
        beta = self.factor_params["beta"]
        sig2_M = self.factor_params["factor_variance"]
        iter_train_items_raitings = iter(zip(train_items, train_ratings))

        tmp_item = []
        tmp_rating = []
        tmp_weight = []
        # initialize
        i_train, r_train = next(iter_train_items_raitings)
        # start
        for i, mu_i, beta_i, sig2_i in zip(range(self.n_items), mu_items, beta, diag_Sigma):
            prev_weight = 1
            prev_rating = 0
            if i == i_train:
                prev_weight = self.alpha
                prev_rating = r_train
                try:
                    i_train, r_train = next(iter_train_items_raitings)
                except StopIteration:
                    i_train = self.n_items
                    r_train = 0
            mv_weight = self.gamma / 2 * self.reg_param_mv * sig2_i
            avg_true_beta = self.avg_true_beta[user] - beta_i / self.num_holding[user] * prev_rating
            mv_rating = (mu_i / self.gamma - (beta_i * np.matmul(avg_true_beta, sig2_M) / 2).sum()) / sig2_i
            weight = prev_weight + mv_weight
            rating = (prev_weight * prev_rating + mv_weight * mv_rating) / weight

            tmp_item.append(i)
            tmp_rating.append(rating)
            tmp_weight.append(weight)
        tmp_item, tmp_rating, tmp_weight = np.array(tmp_item).astype(int), \
                                          np.array(tmp_rating).astype(np.float32), \
                                          np.array(tmp_weight).astype(np.float32)

        new_positive_items = self.filter_user_positive_items(train_items, tmp_item, tmp_rating)

        return tmp_item, tmp_rating, tmp_weight, new_positive_items

    def filter_user_positive_items(self, true_positives, items, ratings):
        if self.positive_score_cri is not None:
            return set(items[ratings > self.positive_score_cri])
        else:
            new_positive_items = set(true_positives)
            num_true = len(true_positives)
            if self.num_pos_to_neg is not None:
                num_delete = int(self.num_pos_to_neg * num_true)
                new_positive_items -= set(true_positives[np.argsort(ratings[true_positives])[:num_delete]])
            if self.num_neg_to_pos is not None:
                num_add = int(self.num_neg_to_pos * num_true)
                modified_ratings = ratings.copy()
                modified_ratings[true_positives] = -np.inf
                new_positive_items = new_positive_items.union(set(items[np.argsort(modified_ratings)[-num_add:]]))
            return new_positive_items

    def _data_processing(self, train, test):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            train (pandas.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pandas.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                test can be None, if so, we only process the training data.

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.

        """
        df = train if test is None else train.append(test)

        if self.user_idx is None:
            user_idx = df[[self.col_user]].drop_duplicates().reindex()
            user_idx[self.col_user + "_idx"] = user_idx[self.col_user]
            self.n_users = len(user_idx)
            self.user_idx = user_idx

            self.user2id = dict(
                zip(user_idx[self.col_user], user_idx[self.col_user + "_idx"])
            )
            self.id2user = dict(
                zip(user_idx[self.col_user + "_idx"], user_idx[self.col_user])
            )

        if self.item_idx is None:
            item_idx = df[[self.col_item]].drop_duplicates()
            item_idx[self.col_item + "_idx"] = item_idx[self.col_item]
            self.n_items = len(item_idx)
            self.item_idx = item_idx

            self.item2id = dict(
                zip(item_idx[self.col_item], item_idx[self.col_item + "_idx"])
            )
            self.id2item = dict(
                zip(item_idx[self.col_item + "_idx"], item_idx[self.col_item])
            )

        return self._reindex(train), self._reindex(test)

    def _reindex(self, df):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating).

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.

        """

        if df is None:
            return None

        df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        df = df[df[self.col_rating] > 0]

        df_reindex = df[
            [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        ]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df_reindex

    def _init_train_data(self):
        """Record items interated with each user in a dataframe self.interact_status, and create adjacency
        matrix self.R.

        """
        self.interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R[self.train[self.col_user], self.train[self.col_item]] = 1.0

    def get_norm_adj_mat(self):
        """Load normalized adjacency matrix if it exists, otherwise create (and save) it.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        try:
            if self.adj_dir is None:
                raise FileNotFoundError
            norm_adj_mat = sp.load_npz(self.adj_dir + "/norm_adj_mat.npz")
            print("Already load norm adj matrix.")

        except FileNotFoundError:
            norm_adj_mat = self.create_norm_adj_mat()
            if self.adj_dir is not None:
                sp.save_npz(self.adj_dir + "/norm_adj_mat.npz", norm_adj_mat)
        return norm_adj_mat

    def create_norm_adj_mat(self):
        """Create normalized adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[: self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("Already create adjacency matrix.")

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)
        print("Already normalize adjacency matrix.")

        return norm_adj_mat.tocsr()

    def sample_neg(self, positive_set: set):
        while True:
            neg_id = random.randint(0, self.n_items - 1)
            if neg_id not in positive_set:
                return neg_id

    def data_loader(self, batch_size):
        """Sample train data every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size (int): Batch size of users.

        Returns:
            numpy.ndarray, numpy.ndarray, numpy.ndarray:
            - Sampled users.
            - Sampled positive items.
            - Sampled negative items.
        """

        indices = range(len(self.interact_status))
        if self.n_users < batch_size:
            users = [random.choice(indices) for _ in range(batch_size)]
        else:
            users = random.sample(indices, batch_size)

        interact = self.interact_status.iloc[users]
        pos_items = interact[self.col_item + "_interacted"].apply(
            lambda x: random.choice(list(x))
        )
        neg_items = interact[self.col_item + "_interacted"].apply(
            lambda x: self.sample_neg(x)
        )

        return np.array(users), np.array(pos_items), np.array(neg_items)
