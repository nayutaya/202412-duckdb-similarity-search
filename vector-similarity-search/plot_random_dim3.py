import duckdb
import matplotlib.pyplot as plt
import numpy as np

# DuckDBからfeature列を取得する
con = duckdb.connect("random_dim3.duckdb")
features = np.vstack(
    con.sql("SELECT feature FROM records").fetchnumpy()["feature"].tolist()
)
print((features.dtype, features.shape))

# 散布図を描画する
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(xs=features[:, 0], ys=features[:, 1], zs=features[:, 2], s=1, alpha=0.7)
plt.tight_layout()
plt.savefig("random_dim3.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
