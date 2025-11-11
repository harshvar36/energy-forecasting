from pathlib import Path

DATA = Path("data")
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

for p in [DATA, DATA_RAW, DATA_PROCESSED]:
    p.mkdir(parents=True, exist_ok=True)

def ensure_parents(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
