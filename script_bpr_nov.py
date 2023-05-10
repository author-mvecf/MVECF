import os
from lib.path import root_path
from lib.utils import get_data, get_correlation
import numpy as np
from lib.models.bpr import BPRnov

holdings_data, factor_params = get_data("CRSP", 2012)

correlation = get_correlation(factor_params)
distance = np.sqrt(1 - correlation)

distance_threshold = 0.9
novelty_rate = 0.8
max_iter = 300000
embedding_dim = 30
lr = 0.005
weight_decay = 0.00001
batch_size = 128

dirpath = os.path.join(root_path, "results/bpr_nov")
print(dirpath)
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

model = BPRnov(
    holdings_data, distance,
    distance_threshold=distance_threshold,
    embedding_dim=embedding_dim,
    lr=lr,
    novelty_rate=novelty_rate,
    weight_decay=weight_decay,
    max_iter=max_iter,
    early_stop=True,
    tmp_save_path=dirpath,
    batch_size=batch_size
)
model.fit()
