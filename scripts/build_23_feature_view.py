import pandas as pd
from pathlib import Path

# required feature names
REQ = ["b","e","AC","FM","UC","DL","DS","DP","DR","LB",
       "ASTV","MSTV","ALTV","MLTV","Width","Min","Max",
       "Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]

src = Path("data/processed/ctg_final.csv")
dst = Path("data/processed/ctg_23train.csv")

if not src.exists():
    raise SystemExit(f"Missing source CSV: {src}")

df = pd.read_csv(src)

# map columns with suffixes back to their base names
rename_map = {}
for col in df.columns:
    base = col.split(".")[0]
    if base in REQ and col != base:
        rename_map[col] = base

df = df.rename(columns=rename_map)

# ensure all required columns exist
for c in REQ:
    if c not in df.columns:
        df[c] = 0

if "NSP" not in df.columns:
    raise SystemExit("Missing NSP label in source CSV.")

# keep only the required features + NSP
view = df[REQ + ["NSP"]].copy()

# basic imputation for safety
view[REQ] = view[REQ].fillna(view[REQ].mean(numeric_only=True))

dst.parent.mkdir(parents=True, exist_ok=True)
view.to_csv(dst, index=False)
print(f"✅ 23-feature training file created: {dst} (rows={len(view)})")
