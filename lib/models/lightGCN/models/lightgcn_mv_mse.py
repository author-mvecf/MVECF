# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
from lib.models.lightGCN.models.lightgcn_mse import LightGCN_MSE
from lib.models.lightGCN.DataModel.ImplicitCF import ImplicitCF
import numpy as np

tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x


class LightGCN_MV_MSE(LightGCN_MSE):
    def __init__(self, hparams, data: ImplicitCF, seed=None, result_dir=None, early_stop=True,
                 gamma=None, reg_param_mv=None):
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        self.gamma = gamma
        self.reg_param_mv = reg_param_mv
        self.data = data
        self.data_laod_func = self.data._train_loader_mv_mse
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
        self.ratings = tf.compat.v1.placeholder(tf.float32, shape=(None,))
        self.loss_weights = tf.compat.v1.placeholder(tf.float32, shape=(None,))

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
