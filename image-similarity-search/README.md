# image-similarity-search

DuckDBを使った類似画像検索の実験コードです。
詳しくは以下の記事をご参照ください。

[Jina CLIP v2とDuckDBを使って類似画像検索してみた](https://zenn.dev/yuyakato/articles/1e585cbc9d97f8)

```sh
python add_image.py --db-file example.duckdb --image-dir example
python search_image.py --db-file example.duckdb --image-file example/pakutaso_30012.jpg
```
