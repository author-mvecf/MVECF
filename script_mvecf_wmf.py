import os
from lib.path import root_path
from lib.utils import get_data
from lib.models.mvecf_wmf import MVECF_WMF

holdings_data, factor_params = get_data("CRSP", 2012)
gamma = 3
latent_dim = 30
lr = 0.001
reg_param = 0.001
reg_param_mv = 10

dirpath = os.path.join(root_path, "results/mvecf_wmf")
print(dirpath)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

mf = MVECF_WMF(
    holdings_data, factor_params, n_epochs=300, n_factors=latent_dim,
    reg_param=reg_param, lr=lr, reg_param_mv=reg_param_mv, gamma=gamma,
    tmp_save_path=dirpath,
)
mf.fit()
