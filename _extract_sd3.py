import pyarrow.parquet as pq
from pathlib import Path

src = Path("data/sd3/data/train-00000-of-00001.parquet")
out_dir = Path("data/sd3/images")
out_dir.mkdir(parents=True, exist_ok=True)

tbl = pq.read_table(src)
n = tbl.num_rows
print("rows:", n)

img_col = tbl.column("image")
written = 0
skipped = 0
for i in range(n):
    rec = img_col[i].as_py()
    data = rec["bytes"]
    if not data:
        skipped += 1
        continue
    p = out_dir / f"sd3_{i:05d}.jpg"
    p.write_bytes(data)
    written += 1

print(f"written={written} skipped={skipped} -> {out_dir}/")
