from pathlib import Path

import click
import duckdb
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


@click.command()
@click.option(
    "--image-dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    help="path to image directory.",
)
@click.option(
    "--db-file",
    required=True,
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, writable=True),
    help="path to DuckDB file.",
)
def main(image_dir, db_file):
    # CLIPモデル・プロセッサのロード
    model_name = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP model ({model_name}) ...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # GPUが使える環境であればGPUを利用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # DuckDBに接続
    print(f"Opening database: {db_file}")
    con = duckdb.connect(db_file)

    # imagesテーブルがなければ作成
    # ここでは 768次元の特徴量を FLOAT[] (配列) として保持する例を示す
    # DuckDBでは厳密にFLOAT[768]と指定できるが、環境によっては一部機能制限があるため
    # 「FLOAT[] NOT NULL」で代用している
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS images (
        file_path VARCHAR NOT NULL PRIMARY KEY,
        feature FLOAT4[768] NOT NULL
    )
    """
    con.execute(create_table_sql)

    # 再帰的に *.jpg を取得
    image_files = list(image_dir.rglob("*.jpg"))
    print(image_files)
    print(f'Found {len(image_files)} images under "{image_dir}"')

    # すでにテーブルに登録済みのファイルパスを一括で取得
    registered_paths = set(
        row[0] for row in con.execute("SELECT file_path FROM images").fetchall()
    )
    print(registered_paths)

    # 画像を1枚ずつ処理
    for idx, img_path in enumerate(image_files, 1):
        if str(img_path) in registered_paths:
            # 既に登録されていればスキップ
            print(f"[{idx}/{len(image_files)}] Skip (already registered): {img_path}")
            continue

        try:
            image = Image.open(img_path).convert("RGB")

            # CLIPの入力を作成
            inputs = processor(images=image, return_tensors="pt")
            # デバイスに転送
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            # 推論
            with torch.no_grad():
                embeddings = model.get_image_features(**inputs)

            # 特徴ベクトルを正規化（任意。検索用途によっては正規化した方が扱いやすい）
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # listに変換 (DuckDBに格納しやすい形式にする)
            feature_vector = embeddings[0].cpu().numpy()

            # データベースにINSERT
            con.execute(
                "INSERT INTO images (file_path, feature) VALUES ($file_path, $feature)",
                {"file_path": str(img_path), "feature": feature_vector},
            )

            print(f"[{idx}/{len(image_files)}] Added: {img_path}")

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] Failed to process {img_path}: {e}")

    # コミットとクローズ
    con.commit()
    con.close()
    print("Done.")


if __name__ == "__main__":
    main()
