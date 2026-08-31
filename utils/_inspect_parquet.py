import pyarrow.parquet as pq

path = "data/sd3/data/train-00000-of-00001.parquet"
pf = pq.ParquetFile(path)
print("num_rows:", pf.metadata.num_rows)
print("num_row_groups:", pf.metadata.num_row_groups)
print("schema:")
for f in pf.schema_arrow:
    print(f"  {f.name}: {f.type}")

# Peek at first row to understand image storage
import pyarrow.dataset as ds
tbl = pq.read_table(path)
print("\ncolumns:", tbl.column_names)
first = tbl.slice(0, 1)
for col in tbl.column_names:
    val = first.column(col)[0].as_py()
    if isinstance(val, (bytes, bytearray)):
        print(f"  {col}: bytes len={len(val)}  head={bytes(val[:16])!r}")
    else:
        s = str(val)
        print(f"  {col}: {type(val).__name__}  {s[:120]!r}")
