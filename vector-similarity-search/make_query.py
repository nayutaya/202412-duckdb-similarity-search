import sys

import numpy as np
from sklearn.preprocessing import normalize

# 次元数をコマンドライン引数から取得する
n_dims = int(sys.argv[1])

# ランダムなクエリベクトルを生成する
query = normalize(np.random.rand(1, n_dims).astype(np.float32) - 0.5)[0]
print((query.dtype, query.shape))

# クエリベクトルを保存する
query_file_name = f"query_dim{n_dims}.npy"
print(query_file_name)
np.save(query_file_name, query)
