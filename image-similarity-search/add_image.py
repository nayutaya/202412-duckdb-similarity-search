from pathlib import Path

import click
import duckdb
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def load_jina_clip_v2(device: torch.device) -> tuple:
    """Jina CLIP v2を読み込む。"""
    model_name = "jinaai/jina-clip-v2"
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


def extract_image_feature(device: torch.device, model, processor, image) -> np.ndarray:
    """画像からCLIP特徴量を取得する。"""
    inputs = processor(images=image, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features /= features.norm(dim=-1, keepdim=True)
    features = features.float().cpu().numpy()
    return features[0]


@click.command()
@click.option(
    "--db-file",
    required=True,
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, writable=True),
    help="path to DuckDB file.",
)
@click.option(
    "--image-dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, exists=True),
    help="path to image directory.",
)
@click.option(
    "--device",
    required=True,
    type=torch.device,
    default=(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    help="device name to use for inference.",
)
def main(db_file: Path, image_dir: Path, device: torch.device) -> None:
    print("Loading model...")
    model, processor = load_jina_clip_v2(device)

    print("Connecting to DB...")
    con = duckdb.connect(db_file)
    con.execute("""
    CREATE TABLE IF NOT EXISTS images(
        file_path VARCHAR NOT NULL PRIMARY KEY,
        feature FLOAT4[1024] NOT NULL
    )
    """)

    print("Finding images...")
    image_files = list(image_dir.rglob("*.jpg"))
    print(f'Found {len(image_files)} images under "{image_dir}"')

    added_paths = set(
        row[0] for row in con.execute("SELECT file_path FROM images").fetchall()
    )

    for index, image_file in enumerate(image_files, 1):
        if str(image_file) in added_paths:
            print(f"[{index}/{len(image_files)}] Skip (already added): {image_file}")
            continue

        try:
            image = Image.open(image_file).convert("RGB")
            feature = extract_image_feature(device, model, processor, image)
            con.execute(
                "INSERT INTO images(file_path, feature) VALUES ($file_path, $feature)",
                {"file_path": str(image_file), "feature": feature},
            )
            print(f"[{index}/{len(image_files)}] Added: {image_file}")
        except Exception as e:
            print(f"[{index}/{len(image_files)}] Error: {image_file}: {e}")

    con.commit()
    con.close()
    print("Done")


if __name__ == "__main__":
    main()
