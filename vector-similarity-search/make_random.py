import os
import sys

import base58
import duckdb
import numpy as np
import polars as pl
from sklearn.preprocessing import normalize

# 次元数、データ数をコマンドライン引数から取得する
n_dims = int(sys.argv[1])
n_records = int(sys.argv[2])

# ランダムなデータを含むNumPy構造体配列を生成する
records_numpy = np.zeros(
    (n_records,), dtype=[("id", "U11"), ("feature", np.float32, (n_dims,))]
)
records_numpy["id"] = np.array(
    [base58.b58encode(os.urandom(8)).decode() for _ in range(n_records)]
)
records_numpy["feature"] = normalize(
    np.random.rand(n_records, n_dims).astype(np.float32) - 0.5
)

# Polarsのデータフレームに変換する
records_df = pl.DataFrame(records_numpy)

# DuckDBに書き込む
db_file_name = f"random_dim{n_dims}.duckdb"
print(db_file_name)
con = duckdb.connect(db_file_name)
con.sql(f"CREATE TABLE records(id VARCHAR PRIMARY KEY, feature FLOAT[{n_dims}])")
con.sql("INSERT INTO records SELECT * FROM records_df")
print(con.sql("SELECT COUNT(*) FROM records"))
