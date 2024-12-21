import duckdb
import numpy as np
import matplotlib.pyplot as plt

# クエリベクトルを読み込む
query = np.load("query_dim3.npy")

# DuckDBでコサイン類似度を算出する
con = duckdb.connect("random_dim3.duckdb")
result = con.execute(
    "SELECT feature, array_cosine_similarity(feature, $query::FLOAT4[3]) AS similarity FROM records",
    {"query": query},
).fetchnumpy()

features = np.vstack(result["feature"])
similarities = result["similarity"]
similarity_mask = similarities >= 0.9

# 結果を可視化する
# * 原点からクエリベクトル点を赤い線として描画する
# * コサイン類似度0.9未満の点を黒い点として描画する
# * コサイン類似度0.9以上の点を類似度で着色し描画する
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot(xs=[0, query[0]], ys=[0, query[1]], zs=[0, query[2]], color="red", linewidth=1)
ax.scatter(
    xs=features[~similarity_mask][:, 0],
    ys=features[~similarity_mask][:, 1],
    zs=features[~similarity_mask][:, 2],
    c="black",
    s=1,
    alpha=0.1,
)
ax.scatter(
    xs=features[similarity_mask][:, 0],
    ys=features[similarity_mask][:, 1],
    zs=features[similarity_mask][:, 2],
    c=similarities[similarity_mask],
    s=2,
    alpha=0.7,
)
plt.tight_layout()
plt.savefig("similarity_dim3.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
