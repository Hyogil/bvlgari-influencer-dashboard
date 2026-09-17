"""Optional helper: export the automatically repaired dataset as UTF-8 CSV."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml import DATA  # noqa: E402

out = Path(__file__).with_name("creators_cleaned_utf8.csv")
DATA.creators.to_csv(out, index=False, encoding="utf-8-sig")
print(f"Saved: {out}")
print(f"Rows: {len(DATA.creators):,}")
print(f"Repaired shifted rows: {DATA.repaired_rows}")
