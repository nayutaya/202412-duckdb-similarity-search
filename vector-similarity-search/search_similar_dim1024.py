import time

import duckdb
import numpy as np

# クエリベクトルを読み込む
n_dims = 1024
query = np.load(f"query_dim{n_dims}.npy")

# DuckDBでコサイン類似度を算出し、上位10レコードを取得する
con = duckdb.connect(f"random_dim{n_dims}.duckdb")
start_time = time.perf_counter()
similar_records = con.execute(
    f"""
    SELECT
      id,
      array_cosine_similarity(feature, $query::FLOAT4[{n_dims}]) AS similarity
    FROM records
    ORDER BY similarity DESC
    LIMIT 10
    """,
    {"query": query},
).fetchall()
end_time = time.perf_counter()

# 検索結果を表示する
for id, similarity in similar_records:
    print(f"{id}: {similarity:.6f}")

print(f"time: {end_time - start_time:.3f} sec")
