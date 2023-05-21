import os
from lib.path import root_path
from lib.utils import get_data
from lib.models.base_svd import SVD_als

holdings_data, factor_params = get_data("CRSP", 2012)
gamma = 3
latent_dim = 30
reg_param = 0.001

dirpath = os.path.join(root_path, "results/wmf")
print(dirpath)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

mf = SVD_als(
    holdings_data, n_epochs=300, n_factors=latent_dim,
    reg_param=reg_param, tmp_save_path=dirpath,
)
mf.fit()
