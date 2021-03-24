from utils.io import _load_cpu
from utils.ddfa import reconstruct_vertex
import pickle
import numpy as np

# params = _load_cpu("train.configs/param_all_norm.pkl")
# lm = reconstruct_vertex(params[0], dense=False, transform=True)

# lms = _load_cpu("train.configs/param_lm_train.pkl")
# lm_ = lms[0][62:]

data = []
params = _load_cpu("train.configs/param_all_norm_val.pkl")

for param in params:
    lm = reconstruct_vertex(param, dense=False, transform=True)
    data.append(np.concatenate((param,lm[:2, :].reshape(-1))))

with open("train.configs/param_lm_val.pkl", "wb") as f:
    pickle.dump(np.array(data), f)