import click
import duckdb


@click.command()
@click.option("--db-file", required=True, type=str, help="DuckDBファイルのパス")
@click.option("--image-file", required=True, type=str, help="検索のクエリとなる画像ファイルパス")
def main(db_file, image_file):
    """
    指定された画像ファイルの特徴量と、データベースに登録されている他画像の特徴量との
    コサイン類似度を求め、上位10件を表示する。
    """

    # DuckDBに接続
    con = duckdb.connect(db_file)

    # 対象画像がDBに登録済みか確認
    check_sql = """
    SELECT COUNT(*) FROM images WHERE file_path = ?
    """
    count = con.execute(check_sql, [image_file]).fetchone()[0]
    if count == 0:
        print(f"Error: file_path='{image_file}' がDBに登録されていません。先にadd_image.pyで登録してください。")
        return

    # 1つのSQLで「コサイン類似度」を求めてソートし上位10件を取得する
    # DuckDBのUNNEST機能＋Window関数を活用して配列同士のドット積とノルムを同時に計算する例
    # ここで HAVING file_path <> ? とすることで、検索元の画像は除外する
    query_sql = f"""
    WITH ref_file_path AS (
        SELECT ? AS path
    ),
    ref AS (
        SELECT UNNEST(feature) AS r_val,
               ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS pos
        FROM images
        WHERE file_path = (SELECT path FROM ref_file_path)
    ),
    unnest_images AS (
        SELECT file_path,
               UNNEST(feature) AS i_val,
               ROW_NUMBER() OVER (PARTITION BY file_path ORDER BY (SELECT NULL)) AS pos
        FROM images
    )
    SELECT
        unnest_images.file_path AS file_path,
        SUM(i_val * r_val) / (SQRT(SUM(i_val * i_val)) * SQRT(SUM(r_val * r_val))) AS similarity
    FROM unnest_images
    JOIN ref ON unnest_images.pos = ref.pos
    GROUP BY unnest_images.file_path
    HAVING file_path <> (SELECT path FROM ref_file_path)
    ORDER BY similarity DESC
    LIMIT 10
    """

    results = con.execute(query_sql, [image_file]).fetchall()

    # 結果を表示
    for rank, (path, sim) in enumerate(results, start=1):
        print(f"{rank}. path={path}, similarity={sim:.4f}")

    con.close()


if __name__ == "__main__":
    main()
