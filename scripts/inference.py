import argparse, json, joblib, pandas as pd
from pathlib import Path

REQ = ["b","e","AC","FM","UC","DL","DS","DP","DR","LB","ASTV","MSTV","ALTV","MLTV",
       "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]

def main():
    ap = argparse.ArgumentParser(description="Run inference from CSV with 23 features.")
    ap.add_argument("--weights", default="models/best_model.joblib")
    ap.add_argument("--input_csv", default="sample_input.csv")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    bundle = joblib.load(args.weights)
    model, feats = bundle["model"], bundle.get("features", REQ)

    X = pd.read_csv(args.input_csv)
    miss = [c for c in REQ if c not in X.columns]
    if miss: raise SystemExit(f"Missing features in input: {miss}")

    X = X[REQ].fillna(X.mean(numeric_only=True))
    try: X = X[feats]
    except KeyError: pass

    pred = model.predict(X)
    out = X.copy(); out["NSP_pred"] = pred
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(json.dumps({"weights": args.weights, "n_inputs": len(X), "out_csv": args.out}, indent=2))

if __name__ == "__main__":
    main()
