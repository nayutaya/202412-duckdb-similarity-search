from pathlib import Path

import click
import duckdb
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


@click.command()
@click.option(
    "--image-dir",
    required=True,
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True, exists=True),
    help="path to image directory.",
)
@click.option(
    "--db-file",
    required=True,
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, writable=True),
    help="path to DuckDB file.",
)
def main(image_dir, db_file):
    print("Loading model...")
    device = torch.device("cuda")
    model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(
        device
    )
    processor = AutoProcessor.from_pretrained(
        "jinaai/jina-clip-v2", trust_remote_code=True
    )

    print("Connecting DB...")
    con = duckdb.connect(db_file)
    con.execute("""
    CREATE TABLE IF NOT EXISTS images (
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
            inputs = processor(images=image, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            with torch.no_grad():
                features = model.get_image_features(**inputs)

            features /= features.norm(dim=-1, keepdim=True)
            features = features[0].float().cpu().numpy()

            con.execute(
                "INSERT INTO images(file_path, feature) VALUES ($file_path, $feature)",
                {"file_path": str(image_file), "feature": features},
            )

            print(f"[{index}/{len(image_files)}] Added: {image_file}")

        except Exception as e:
            print(f"[{index}/{len(image_files)}] Failed to process: {image_file}: {e}")

    con.commit()
    con.close()
    print("Done")


if __name__ == "__main__":
    main()
