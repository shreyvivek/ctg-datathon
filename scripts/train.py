import argparse, json, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score

REQ = ["b","e","AC","FM","UC","DL","DS","DP","DR","LB","ASTV","MSTV","ALTV","MLTV",
       "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance","Tendency"]

def main():
    ap = argparse.ArgumentParser(description="Train RF on CTG (23 required features).")
    ap.add_argument("--train_csv", required=True, help="CSV with 23 features + NSP label")
    ap.add_argument("--label_col", default="NSP")
    ap.add_argument("--out", default="models/best_model.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    miss = [c for c in REQ if c not in df.columns]
    if miss: raise SystemExit(f"Missing features: {miss}")
    if args.label_col not in df.columns: raise SystemExit(f"Missing label col: {args.label_col}")

    X = df[REQ].fillna(df[REQ].mean(numeric_only=True))
    y = df[args.label_col].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=42)
    clf.fit(Xtr, ytr)

    pred = clf.predict(Xte)
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro"))
    }
    joblib.dump({"model": clf, "features": REQ}, args.out)
    print(json.dumps({"weights": args.out, **metrics}, indent=2))

if __name__ == "__main__":
    main()
