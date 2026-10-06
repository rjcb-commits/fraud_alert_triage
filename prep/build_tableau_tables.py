"""Step 3: Tableau-ready tables for the fraud alert triage dashboard.

Outputs (tableau/):
  queue_curve.csv          precision and fraud caught at every review capacity (alert score vs random)
  score_bands.csv          alerts by risk-score band: volume, fraud rate, share of all fraud
  signal_lift.csv          fraud-signal lift (copied from step 1)
  analyst_accuracy.csv     each synthetic analyst: accuracy, fraud catch rate, false-alarm rate
  triage_cases.csv         AI triage results per case (priority, status, human decision, actual)
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"; RAW = ROOT / "data_raw" / "FiFAR"
T = ROOT / "tableau"; T.mkdir(exist_ok=True)
a = pd.read_parquet(OUT / "alerts_scored.parquet")
y = a["fraud_bool"].values; n_fraud = y.sum()

# queue curve
order = np.argsort(-a["model_score"].values)
cum = np.cumsum(y[order])
caps = sorted(set([10, 25, 50] + list(range(100, 2001, 100)) + list(range(2500, len(a) + 1, 2500)) + [len(a)]))
pd.DataFrame([{"review_capacity": c, "precision_score_order": round(cum[c - 1] / c, 4),
               "fraud_caught_pct": round(cum[c - 1] / n_fraud, 4), "precision_random": round(y.mean(), 4),
               "fraud_caught_random_pct": round(c / len(a), 4)} for c in caps]).to_csv(T / "queue_curve.csv", index=False)

# score bands
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
a["score_band"] = pd.cut(a["model_score"], bins, right=False, labels=[f"{b:.1f}-{min(bins[i+1],1):.1f}" for i, b in enumerate(bins[:-1])])
sb = a.groupby("score_band", observed=True).agg(alerts=("fraud_bool", "size"), fraud=("fraud_bool", "sum")).reset_index()
sb["fraud_rate"] = (sb["fraud"] / sb["alerts"]).round(4); sb["share_of_all_fraud"] = (sb["fraud"] / n_fraud).round(4)
sb.to_csv(T / "score_bands.csv", index=False)

# signal lift
pd.read_csv(OUT / "signal_lift.csv").to_csv(T / "signal_lift.csv", index=False)

# analysts
e = pd.read_parquet(RAW / "synthetic_experts" / "expert_predictions.parquet")
if e.index.name == "case_id": e = e.reindex(a["case_id"].values)
e = e.reset_index(drop=True)
rows = []
for c in e.columns:
    p = e[c].values
    rows.append({"analyst": c, "accuracy": round((p == y).mean(), 4),
                 "fraud_catch_rate": round(((p == 1) & (y == 1)).sum() / n_fraud, 4),
                 "false_alarm_rate": round(((p == 1) & (y == 0)).sum() / (y == 0).sum(), 4)})
pd.DataFrame(rows).to_csv(T / "analyst_accuracy.csv", index=False)

# triage results
tr = OUT / "triage_audit_log.jsonl"
if tr.exists():
    recs = []
    for r in map(json.loads, open(tr)):
        o = r["assistant_output"]
        recs.append({"case_id": r["input"]["case_id"], "sample": r["sample"], "risk_score": r["input"]["risk_score"],
                     "queue_rank": r["input"]["queue_rank"], "priority_from_score": o["priority"],
                     "status": r["status"], "attempts": r["attempts"], "latency_s": r["latency_s"],
                     "human_decision": r["human_decision"]["decision"], "actual": "FRAUD" if r["ground_truth"] else "LEGITIMATE",
                     "summary": o["summary"], "investigator_note": o.get("investigator_note", ""), "signals": "; ".join(o["risk_signals"]), "checks": "; ".join(o["suggested_checks"])})
    pd.DataFrame(recs).to_csv(T / "triage_cases.csv", index=False)
print(sorted(p.name for p in T.iterdir()))
print(sb.to_string(index=False))
