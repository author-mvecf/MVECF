import numpy as np
from lib.models.base_svd import SVD
from sklearn.metrics import average_precision_score
import os
from joblib import Parallel, delayed

os.environ['OPENBLAS_NUM_THREADS'] = '4'
__all__ = [
    "MVECF_WMF",
]


class MVECF_WMF(SVD):
    def __init__(self, data, factor_params, n_factors=30, n_epochs=100, lr=0.001, reg_param=0.001, reg_param2=0.001,
                 early_stop=True, alpha=10, batch_size=128, reg_param_mv=None, gamma=None,
                 tmp_save_path=None, tmp_load_path=None, ):
        super(MVECF_WMF, self).__init__(data=data, n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg_param=reg_param,
                                        early_stop=early_stop, alpha=alpha, batch_size=batch_size,
                                        tmp_save_path=tmp_save_path, tmp_load_path=tmp_load_path, )
        self.factor_params = factor_params

        self.reg_param_mv = reg_param_mv
        if reg_param2 is None:
            reg_param2 = self.reg_param
        self.reg_param2 = reg_param2
        self.gamma = gamma
        self.val_loss_mv = 0
        beta = self.factor_params["beta"]
        self.avg_true_beta = []
        for user in range(self.n_users):
            train_items = self.train_data[1][self.train_indptr[user]:self.train_indptr[user + 1]]
            self.avg_true_beta.append(beta[train_items].mean(axis=0))
        self.avg_true_beta = np.array(self.avg_true_beta)

        sig2_M = self.factor_params["factor_variance"]
        beta = self.factor_params["beta"]
        sig2_eps = self.factor_params["sig_eps_square"]
        Sigma = np.matmul(np.matmul(beta, sig2_M), beta.T) + np.diag(sig2_eps)
        self.diag_Sigma = np.diag(Sigma)
        self.off_diag_Sigma = Sigma - np.diag(self.diag_Sigma)

        # generate dense rating and weight
        self.dense_rating = []
        self.dense_weight = []
        self.num_holding = []
        for user in range(self.n_users):
            index_start = self.train_indptr[user]
            index_end = self.train_indptr[user + 1]
            self.num_holding.append(index_end - index_start)
            train_items = np.append(self.train_data[1][index_start:index_end], self.n_items + 1)
            train_ratings = np.append(self.train_data[2][index_start:index_end], 0)
            i_user, r_user, w_user = self._get_total_items_ratings_init(user, train_items, train_ratings)

            self.dense_rating.append(r_user)
            self.dense_weight.append(w_user)
        self.num_holding = np.array(self.num_holding)
        self.dense_rating = np.array(self.dense_rating)
        self.dense_rating = self.dense_rating - self.dense_rating.mean()
        self.dense_weight = np.array(self.dense_weight)
        self.params_not_save += [
            "factor_params", "diag_Sigma", "off_diag_Sigma", "sum_of_true_beta", "dense_rating", "dense_weight",
            "num_holding"
        ]

    def _get_total_items_ratings_init(self, user, train_items, train_ratings):
        beta = self.factor_params["beta"]
        mu_M = self.factor_params["factor_mean"]
        sig2_M = self.factor_params["factor_variance"]
        iter_train_items_raitings = iter(zip(train_items, train_ratings))

        tmp_item = []
        tmp_rating = []
        tmp_weight = []
        # initialize
        i_train, r_train = next(iter_train_items_raitings)
        # start
        for i, beta_i, sig2_i in zip(range(self.n_items), beta, self.diag_Sigma):
            prev_weight = 1
            prev_rating = 0
            if i == i_train:
                prev_weight = self.alpha
                prev_rating = r_train
                i_train, r_train = next(iter_train_items_raitings)

            mv_weight = self.gamma / 2 * self.reg_param_mv * sig2_i
            avg_true_beta = self.avg_true_beta[user] - beta_i / self.num_holding[user] * prev_rating
            mv_rating = (beta_i * (mu_M / self.gamma - np.matmul(avg_true_beta, sig2_M) / 2)).sum() / sig2_i
            weight = prev_weight + mv_weight
            rating = (prev_weight * prev_rating + mv_weight * mv_rating) / weight
            if np.isnan(rating):
                a = -1
            tmp_item.append(i)
            tmp_rating.append(rating)
            tmp_weight.append(weight)

        return np.array(tmp_item).astype(int), np.array(tmp_rating).astype(np.float32), np.array(tmp_weight).astype(
            np.float32)

    def _get_total_items_ratings(self, user, train_items, train_ratings):
        return np.arange(self.n_items), self.dense_rating[user], self.dense_weight[user]

    def calculate_valloss(self):
        print("calculating validation loss")
        # rec loss (does not affect early stopping)
        u_total = self.valid_data[0]
        i_total = self.valid_data[1]
        r_total = self.valid_data[2]
        w_total = (1 - r_total) + r_total * self.alpha
        pud = self.pu[u_total]
        qid = self.qi[i_total]
        r_predict = np.sum(pud * qid, axis=1)
        self.val_loss_rec = (w_total * (r_total - r_predict) ** 2).sum()

        # mv loss (does not affect early stopping)
        mu_M = self.factor_params["factor_mean"]
        beta = self.factor_params["beta"]
        beta_i = beta[i_total]
        mu_i = (beta_i * mu_M).sum(axis=1, keepdims=True)

        estimate_ui = r_predict.reshape(-1, 1)
        loss = self.gamma / 2 * (
                estimate_ui ** 2 * self.diag_Sigma[i_total].reshape(-1, 1)
                + estimate_ui * (pud * np.matmul(self.off_diag_Sigma, self.qi)[i_total]).sum(axis=1, keepdims=True)
        ) - mu_i * estimate_ui
        self.val_loss_mv = loss.sum()

        # validation
        beta = self.factor_params["beta"]
        mu_M = self.factor_params["factor_mean"]
        sig2_M = self.factor_params["factor_variance"]

        self.val_loss = 0
        self.map_valid = 0
        for user in range(self.n_users):
            index = range(self.valid_indptr[user], self.valid_indptr[user + 1])
            items = self.valid_data[1][index]

            beta_i = beta[items]
            sig2_i = self.diag_Sigma[items].reshape(-1, 1)

            prev_rating = self.valid_data[2][index].reshape(-1, 1)
            prev_weight = (1 - prev_rating) + prev_rating * self.alpha

            mv_weight = self.gamma / 2 * self.reg_param_mv * sig2_i
            mv_rating = (
                                beta_i * (mu_M / self.gamma - np.matmul(self.avg_true_beta[user], sig2_M) / 2)
                        ).sum(axis=1, keepdims=True) / sig2_i
            weight = prev_weight + mv_weight
            rating = (prev_weight * prev_rating + mv_weight * mv_rating) / weight

            weight = weight.flatten()
            rating = rating.flatten()
            # weight = np.clip(weight, 0, self.base_weight * self.alpha)
            estimate = np.matmul(self.pu[user], self.qi[items].T)
            self.val_loss += (weight * (rating - estimate) ** 2).sum()

            y_true = prev_rating
            y_score = estimate

            self.map_valid += average_precision_score(y_true, y_score)

        self.map_valid = self.map_valid / self.n_users
        print("validation MAP: {}".format(self.map_valid))

    def fit_epoch(self, shuffle=True):
        lambda_pu_reg = self.reg_param
        lambda_qi_reg = self.reg_param2
        dtype = 'float32'

        S = self.dense_weight - 1

        self.pu = recompute_factors_batched(self.qi, S, self.dense_rating, lambda_pu_reg, dtype=dtype)
        self.qi = recompute_factors_batched(self.pu, S.T, self.dense_rating.T, lambda_qi_reg, dtype=dtype)


def solve_sequential(As, Bs):
    X_stack = np.empty_like(As, dtype=As.dtype)

    for k in range(As.shape[0]):
        X_stack[k] = np.linalg.solve(Bs[k], As[k])

    return X_stack


def solve_batch(b, S, Y, X_reg, YTYpR, rating, batch_size, m, f, dtype):
    lo = b * batch_size
    hi = min((b + 1) * batch_size, m)
    current_batch_size = hi - lo

    A_stack = np.empty((current_batch_size, f), dtype=dtype)
    B_stack = np.empty((current_batch_size, f, f), dtype=dtype)

    for ib, k in enumerate(range(lo, hi)):
        s_u = S[k]
        p_u = rating[k]
        A = ((s_u + 1) * p_u).dot(Y)  # YTCup(u)

        if X_reg is not None:
            A += X_reg[k]

        YTSY = np.dot(Y.T, (Y * s_u[:, None]))
        B = YTSY + YTYpR

        A_stack[ib] = A
        B_stack[ib] = B
    # solve B_stack * X = A_stack
    X_stack = solve_sequential(A_stack, B_stack)
    return X_stack


def recompute_factors_batched(Y, S, rating, lambda_reg, X=None,
                              dtype='float32', batch_size=2000, n_jobs=20):
    m = S.shape[0]  # m = number of users
    f = Y.shape[1]  # f = number of factors

    YTY = np.dot(Y.T, Y)  # precompute this
    YTYpR = YTY + lambda_reg * np.eye(f)
    if X is not None:
        X_reg = lambda_reg * X
    else:
        X_reg = None
    X_new = np.zeros((m, f), dtype=dtype)

    num_batches = int(np.ceil(m / float(batch_size)))

    res = Parallel(n_jobs=n_jobs)(delayed(solve_batch)(b, S, Y, X_reg, YTYpR, rating,
                                                       batch_size, m, f, dtype)
                                  for b in range(num_batches))
    X_new = np.concatenate(res, axis=0)

    return X_new
