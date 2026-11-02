"""Read and write the turn CSVs exactly as view_data.export wrote them."""
import csv
import time
from pathlib import Path


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames), list(reader)


def write(path: Path, fields, rows):
    """Write to a temp file, then replace, so a crash never leaves a half-written CSV."""
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for attempt in range(20):   # OneDrive, an editor or a virus scan can hold the target for a moment
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.5)


def key(row):
    return f"{row['dialogue']}:{row['turn']}"
