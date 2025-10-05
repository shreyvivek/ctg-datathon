import argparse, json, joblib, pandas as pd
from pathlib import Path

# Required feature columns (23)
REQ = [
    "b","e","AC","FM","UC","DL","DS","DP","DR",
    "LB","ASTV","MSTV","ALTV","MLTV",
    "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/best_model.joblib")
    ap.add_argument("--input_csv", default="sample_input.csv")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    # Load model
    bundle = joblib.load(args.weights)
    model = bundle["model"]
    feats = bundle.get("features", REQ)

    # Load input CSV
    X = pd.read_csv(args.input_csv)

    # Validate features
    missing = [c for c in REQ if c not in X.columns]
    if missing:
        raise SystemExit(f"❌ Missing required features: {missing}")

    # Select and clean feature columns
    X = X[REQ].fillna(X.mean(numeric_only=True))
    try:
        X = X[feats]
    except KeyError:
        pass

    # Predict NSP
    preds = model.predict(X)
    out = X.copy()
    out["NSP_pred"] = preds

    # Save output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(json.dumps({
        "weights": args.weights,
        "n_inputs": len(X),
        "out_csv": args.out
    }, indent=2))

if __name__ == "__main__":
    main()

