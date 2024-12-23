from pathlib import Path

import click
import duckdb


@click.command()
@click.option(
    "--image-file",
    required=True,
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False, exists=True),
    help="path to query image",
)
@click.option(
    "--db-file",
    required=True,
    type=click.Path(
        path_type=Path, file_okay=True, dir_okay=False, writable=True, exists=True
    ),
    help="path to DuckDB file.",
)
def main(db_file, image_file):
    con = duckdb.connect(db_file)

    count = con.execute(
        "SELECT COUNT(*) FROM images WHERE file_path = ?", (str(image_file),)
    ).fetchone()[0]
    if count == 0:
        print(f'Error: "{image_file}" was not found in DB')
        return

    images = con.execute(
        """
        SELECT
            a.file_path,
            array_cosine_similarity(a.feature, b.feature) AS similarity
        FROM images AS a, images AS b
        WHERE b.file_path = $query_file_path AND a.file_path <> $query_file_path
        ORDER BY similarity DESC
        LIMIT 10
        """,
        {"query_file_path": str(image_file)},
    ).fetchall()

    for file_path, similarity in images:
        print(f"{file_path}: {similarity:.04f}")

    con.close()


if __name__ == "__main__":
    main()
