import os
import pandas as pd
import tensorflow as tf
from lib.utils import get_data
from lib.path import root_path
from lib.models.lightGCN.utils.timer import Timer
from lib.models.lightGCN.models.lightgcn import LightGCN
from lib.models.lightGCN.DataModel.ImplicitCF import ImplicitCF
from lib.models.lightGCN.utils.constants import SEED as DEFAULT_SEED
from lib.models.lightGCN.utils.deeprec_utils import prepare_hparams

tf.get_logger().setLevel('ERROR')  # only show error messages

data_type = "CRSP"
target_year = 2012

# Model parameters
lr = 0.005
TOP_K = 20
EPOCHS = 300
BATCH_SIZE = 128
SEED = DEFAULT_SEED  # Set None for non-deterministic results

gamma = 3
reg_param_mv = 10

positive_score_cri = 0.195  # for CRSP 2012 dataset

dirpath = os.path.join(root_path, f"results/mvecf_lgcn")
holdings_data, factor_params = get_data(data_type, target_year)
yaml_file = os.path.join(root_path, "lib/models/lightGCN/config/lightgcn.yaml")

train = pd.DataFrame(holdings_data["train_data"].T, columns=["userID", "itemID", "rating"])
test = pd.DataFrame(holdings_data["valid_data"].T, columns=["userID", "itemID", "rating"])

# data = ImplicitCF(train=train, test=test, seed=SEED) for ordinary lightGCN
data = ImplicitCF(train=train, test=test, seed=SEED,
                  factor_params=factor_params, reg_param_mv=reg_param_mv, gamma=gamma,
                  positive_score_cri=positive_score_cri)

hparams = prepare_hparams(yaml_file,
                          n_layers=4,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          learning_rate=lr,
                          eval_epoch=1,
                          top_k=TOP_K,
                          save_model=True,
                          save_epoch=1,
                          embed_size=30,
                          )

model = LightGCN(hparams, data, seed=SEED, result_dir=dirpath, gamma=gamma, reg_param_mv=reg_param_mv,
                 early_stop=False)

with Timer() as train_time:
    model.fit()
model.sess.close()
print("Took {} seconds for training.".format(train_time.interval))
