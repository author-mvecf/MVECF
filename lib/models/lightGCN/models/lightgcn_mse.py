# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import time
import os
import sys
import numpy as np
from lib.models.lightGCN.models.lightgcn import LightGCN
from lib.models.lightGCN.DataModel.ImplicitCF import ImplicitCF
from lib.utils import save_pickle

tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x


class LightGCN_MSE(LightGCN):
    def __init__(self, hparams, data: ImplicitCF, seed=None, result_dir=None, early_stop=True,
                 gamma=None, reg_param_mv=None):
        """Initializing the model. Create parameters, placeholders, embeddings and loss function.

        Args:
            hparams (HParams): A HParams object, hold the entire set of hyperparameters.
            data (object): A recommenders.DataModel.ImplicitCF object, load and process data.
            seed (int): Seed.

        """

        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.gamma = gamma
        self.reg_param_mv = reg_param_mv
        self.data = data
        self.data_laod_func = self.data._train_loader_mse
        self.epochs = hparams.epochs
        self.lr = hparams.learning_rate
        self.emb_dim = hparams.embed_size
        self.batch_size = hparams.batch_size
        self.n_layers = hparams.n_layers
        self.decay = hparams.decay
        self.eval_epoch = hparams.eval_epoch
        self.top_k = hparams.top_k
        # self.save_model = hparams.save_model
        # self.save_epoch = hparams.save_epoch
        self.metrics = hparams.metrics
        # self.model_dir = hparams.model_dir
        self.list_stop_cri = []
        self.result_dir = result_dir
        self.early_stop = early_stop

        metric_options = ["map", "ndcg", "precision", "recall"]
        for metric in self.metrics:
            if metric not in metric_options:
                raise ValueError(
                    "Wrong metric(s), please select one of this list: {}".format(
                        metric_options
                    )
                )

        self.norm_adj = data.get_norm_adj_mat()

        self.n_users = data.n_users
        self.n_items = data.n_items

        self.users = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.items = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.ratings = tf.compat.v1.placeholder(tf.int32, shape=(None,))
        self.loss_weights = tf.compat.v1.placeholder(tf.int32, shape=(None,))

        self.weights = self._init_weights()
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        self.u_g_embeddings = tf.nn.embedding_lookup(
            params=self.ua_embeddings, ids=self.users
        )
        self.i_g_embeddings = tf.nn.embedding_lookup(
            params=self.ia_embeddings, ids=self.items
        )
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["user_embedding"], ids=self.users
        )
        self.i_g_embeddings_pre = tf.nn.embedding_lookup(
            params=self.weights["item_embedding"], ids=self.items
        )

        self.batch_ratings = tf.matmul(
            self.u_g_embeddings,
            self.i_g_embeddings,
            transpose_a=False,
            transpose_b=True,
        )

        self.mf_loss, self.emb_loss = self._create_mse_loss(
            self.u_g_embeddings, self.i_g_embeddings, self.ratings, self.loss_weights
        )
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss
        )
        # self.saver = tf.compat.v1.train.Saver(max_to_keep=5)

        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        )
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def _create_mse_loss(self, users, items, ratings, loss_weights):
        """Calculate MSE loss.

        Args:
            users (tf.Tensor): User embeddings to calculate loss.
            items (tf.Tensor): Positive item embeddings to calculate loss.
            ratings (tf.Tensor): Negative item embeddings to calculate loss.

        Returns:
            tf.Tensor, tf.Tensor: Matrix factorization loss. Embedding regularization loss.

        """
        item_scores = tf.reduce_sum(input_tensor=tf.multiply(users, items), axis=1)
        regularizer = (
                tf.nn.l2_loss(self.u_g_embeddings_pre)
                + tf.nn.l2_loss(self.i_g_embeddings_pre)
        )
        regularizer = regularizer / self.batch_size
        mf_loss = tf.compat.v1.losses.mean_squared_error(ratings, item_scores, loss_weights)
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss

    def fit(self):
        """Fit the model on self.data.train. If eval_epoch is not -1, evaluate the model on `self.data.test`
        every `eval_epoch` epoch to observe the training status.

        """
        for epoch in range(1, self.epochs + 1):
            train_start = time.time()
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = self.data.train.shape[0] // self.batch_size + 1
            for idx in range(n_batch):
                users, items, ratings, loss_weights = self.data_laod_func(self.batch_size)
                _, batch_loss, batch_mf_loss, batch_emb_loss = self.sess.run(
                    [self.opt, self.loss, self.mf_loss, self.emb_loss],
                    feed_dict={
                        self.users: users,
                        self.items: items,
                        self.ratings: ratings,
                        self.loss_weights: loss_weights,
                    },
                )
                loss += batch_loss / n_batch
                mf_loss += batch_mf_loss / n_batch
                emb_loss += batch_emb_loss / n_batch

            train_end = time.time()
            train_time = train_end - train_start
            self.print_and_save_results(epoch, train_time, loss, mf_loss, emb_loss)
            if self.early_stop and self.check_early_stop():
                break
