import numpy as np
from copy import deepcopy
from lib.models.base_svd import SVD

__all__ = [
    "MVECF_reg",
]


class MVECF_reg(SVD):
    def __init__(self, data, factor_params, n_factors=30, n_epochs=100, lr=0.001,
                 reg_param=0.001, early_stop=True, alpha=10, batch_size=128, reg_param_mv=None, gamma=None,
                 tmp_save_path=None, tmp_load_path=None, ):
        super(MVECF_reg, self).__init__(data=data, n_factors=n_factors, n_epochs=n_epochs, lr=lr, reg_param=reg_param,
                                        early_stop=early_stop, alpha=alpha, batch_size=batch_size,
                                        tmp_save_path=tmp_save_path, tmp_load_path=tmp_load_path, )
        self.factor_params = factor_params
        if reg_param_mv is None:
            self.reg_param_mv = self.reg_param
        else:
            self.reg_param_mv = reg_param_mv

        self.gamma = gamma
        self.val_loss_mv = 0

        self.num_holding = []
        for user in range(self.n_users):
            index_start = self.train_indptr[user]
            index_end = self.train_indptr[user + 1]
            self.num_holding.append(index_end - index_start)
        self.num_holding = np.array(self.num_holding).reshape(-1, 1)

        sig2_M = self.factor_params["factor_variance"]
        beta = self.factor_params["beta"]
        sig2_eps = self.factor_params["sig_eps_square"]
        Sigma = np.matmul(np.matmul(beta, sig2_M), beta.T) + np.diag(sig2_eps)
        self.diag_Sigma = np.diag(Sigma)
        self.off_diag_Sigma = Sigma - np.diag(self.diag_Sigma)

        self.params_not_save += ["factor_params", "diag_Sigma", "off_diag_Sigma", "num_holding"]

    def update(self, user, item, rating, weight):
        pud = deepcopy(self.pu[user])
        qid = deepcopy(self.qi[item])
        estimate_ui = np.sum(pud * qid, axis=1)
        err = rating - estimate_ui

        mu_M = self.factor_params["factor_mean"]
        sig2_M = self.factor_params["factor_variance"]

        beta = self.factor_params["beta"]
        sig2_eps = self.factor_params["sig_eps_square"]

        beta_q = np.matmul(beta.T, self.qi)  # BTq
        VBTQ = np.matmul(sig2_M, beta_q)
        sig2_epsQ = self.qi * sig2_eps.reshape(-1, 1)

        beta_i = beta[item]
        mu_i = (beta_i * mu_M).sum(axis=1, keepdims=True)
        Sigma_i_T_Q = np.matmul(beta_i, VBTQ) + sig2_epsQ[item]  # batchsize by numfactor
        pu_QT_Sigma_i = (pud * Sigma_i_T_Q).sum(axis=1, keepdims=True)

        grad_mv_by_pud = self.gamma / self.num_holding[user] / 2 * (
                pu_QT_Sigma_i * qid + estimate_ui.reshape(-1, 1) * Sigma_i_T_Q
        ) - mu_i * qid
        grad_mv_by_qid = self.gamma / self.num_holding[user] * pu_QT_Sigma_i * pud - mu_i * pud

        # update factors
        self.pu[user] += self.lr * (
                (weight * err).reshape((-1, 1)) * qid - self.reg_param * pud - self.reg_param_mv * grad_mv_by_pud
        )
        self.qi[item] += self.lr * (
                (weight * err).reshape((-1, 1)) * pud - self.reg_param * qid - self.reg_param_mv * grad_mv_by_qid
        )
        self.train_loss_rec += (weight * err ** 2).sum()

    def calculate_valloss(self):
        pud, qid, estimate_ui = super(MVECF_reg, self).calculate_valloss()
        user = self.valid_data[0]
        item = self.valid_data[1]
        mu_M = self.factor_params["factor_mean"]
        beta = self.factor_params["beta"]
        beta_i = beta[item]
        mu_i = (beta_i * mu_M).sum(axis=1, keepdims=True)

        estimate_ui = estimate_ui.reshape(-1, 1)
        loss = self.gamma / self.num_holding[user] / 2 * (
                estimate_ui ** 2 * self.diag_Sigma[item].reshape(-1, 1)
                + estimate_ui * (pud * np.matmul(self.off_diag_Sigma, self.qi)[item]).sum(axis=1, keepdims=True)
        ) - mu_i * estimate_ui

        self.val_loss_mv = loss.sum()
        self.val_loss += self.reg_param_mv * self.val_loss_mv
