"""Step 1 of the fraud alert triage project: score alerts and generate reason codes.

Data: FiFAR (Financial Fraud Alert Review, CC BY), built on Feedzai's Bank Account Fraud (BAF)
NeurIPS 2022 dataset: 1M synthetic bank-account-opening applications; 30,622 of them were flagged
by an alert model and reviewed by 50 synthetic fraud analysts.

What this does
  1. Trains an XGBoost model on applications from months 0-2 (never on the alert months).
  2. Scores every alert and computes per-case reason codes from XGBoost feature contributions
     (pred_contribs, i.e. TreeSHAP), translated into investigator-readable phrases.
  3. Compares review-queue precision at fixed daily capacity: alert-model order, our model,
     random order, and the average synthetic analyst.

Outputs (output/): alerts_scored.parquet, queue_metrics.csv, signal_lift.csv, model_card.md
"""
import json, pathlib
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw" / "FiFAR"
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
CAT = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

base = pd.read_csv(RAW / "alert_data" / "Base.csv")
alerts = pd.read_parquet(RAW / "alert_data" / "processed_data" / "alerts.parquet")
experts = pd.read_parquet(RAW / "synthetic_experts" / "expert_predictions.parquet")
if alerts.index.name == "case_id":                      # case_id is stored as the index
    experts = experts.reindex(alerts.index)
    alerts = alerts.reset_index()
    experts = experts.reset_index(drop=True)
assert experts.notna().all().all() and len(experts) == len(alerts)
alert_months = sorted(alerts["month"].unique().tolist())
train = base[base["month"] < min(alert_months)].copy()
feats = [c for c in base.columns if c not in ("fraud_bool", "month")]

def prep(df):
    X = df[feats].copy()
    for c in CAT: X[c] = X[c].astype("category")
    return X

cat_levels = {c: sorted(base[c].astype(str).unique()) for c in CAT}
def prep_fixed(df):
    X = df[feats].copy()
    for c in CAT: X[c] = pd.Categorical(X[c].astype(str), categories=cat_levels[c])
    return X

Xtr, ytr = prep_fixed(train), train["fraud_bool"]
model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(len(ytr) - ytr.sum()) / ytr.sum(), enable_categorical=True,
                          tree_method="hist", eval_metric="aucpr", random_state=7)
model.fit(Xtr, ytr)

Xa = prep_fixed(alerts)
alerts["our_score"] = model.predict_proba(Xa)[:, 1]
contrib = model.get_booster().predict(xgb.DMatrix(Xa, enable_categorical=True), pred_contribs=True)[:, :-1]

# investigator-readable reason phrases
def phrase(f, row):
    v = row[f]
    m = {
        "name_email_similarity": lambda: f"name and email don't match (similarity {v:.2f})",
        "device_distinct_emails_8w": lambda: f"device used with {int(v)} different emails in 8 weeks",
        "device_fraud_count": lambda: f"device previously tied to {int(v)} fraud case(s)",
        "date_of_birth_distinct_emails_4w": lambda: f"date of birth seen with {int(v)} emails in 4 weeks",
        "phone_home_valid": lambda: "home phone not valid" if v == 0 else "home phone valid",
        "phone_mobile_valid": lambda: "mobile phone not valid" if v == 0 else "mobile phone valid",
        "email_is_free": lambda: "free email provider" if v == 1 else "non-free email domain",
        "keep_alive_session": lambda: "did not keep session alive" if v == 0 else "kept session alive",
        "foreign_request": lambda: "request from a foreign IP" if v == 1 else "domestic request",
        "prev_address_months_count": lambda: "no prior address on file" if v == -1 else f"{int(v)} months at prior address",
        "current_address_months_count": lambda: f"{int(v)} months at current address",
        "bank_months_count": lambda: "no prior bank relationship" if v == -1 else f"{int(v)} months with prior bank",
        "credit_risk_score": lambda: f"credit risk score {int(v)}",
        "proposed_credit_limit": lambda: f"proposed credit limit ${int(v):,}",
        "customer_age": lambda: f"applicant age band {int(v)}",
        "income": lambda: f"income decile {v:.1f}",
        "velocity_6h": lambda: f"{v:,.0f} applications in the last 6 hours (velocity)",
        "velocity_24h": lambda: f"{v:,.0f} applications in the last 24 hours (velocity)",
        "velocity_4w": lambda: f"{v:,.0f} applications in the last 4 weeks (velocity)",
        "zip_count_4w": lambda: f"{int(v):,} applications from same ZIP in 4 weeks",
        "session_length_in_minutes": lambda: f"session length {v:.1f} min",
        "has_other_cards": lambda: "has other cards with the bank" if v == 1 else "no other cards with the bank",
        "intended_balcon_amount": lambda: f"intended initial deposit {v:.1f}",
        "days_since_request": lambda: f"{v:.2f} days since request",
        "bank_branch_count_8w": lambda: f"{int(v)} applications at branch in 8 weeks",
        "housing_status": lambda: f"housing status {v}", "employment_status": f"employment status {v}",
        "payment_type": lambda: f"payment plan {v}", "source": f"applied via {v}", "device_os": f"device OS {v}",
    }
    fn = m.get(f)
    return (fn() if callable(fn) else fn) if fn is not None else f"{f} = {v}"

reasons = []
for i in range(len(alerts)):
    row = alerts.iloc[i]
    top = np.argsort(-contrib[i])[:3]
    reasons.append([{"feature": feats[k], "contribution": round(float(contrib[i][k]), 3), "text": phrase(feats[k], row)}
                    for k in top if contrib[i][k] > 0])
alerts["reasons"] = [json.dumps(r) for r in reasons]

# expert accuracy and review-queue precision at fixed capacity
exp_acc = (experts.values == alerts[["fraud_bool"]].values).mean(axis=0)
exp_tpr = [((experts[c] == 1) & (alerts["fraud_bool"] == 1)).sum() / alerts["fraud_bool"].sum() for c in experts.columns]
alerts["analyst_fraud_votes"] = experts.sum(axis=1).values
y = alerts["fraud_bool"]
rows = []
for cap in (100, 250, 500, 1000, 2000):
    def prec(order): return y.loc[order[:cap]].mean()
    rows.append({"daily_capacity": cap,
                 "alert_model_order": round(prec(alerts["model_score"].sort_values(ascending=False).index), 4),
                 "our_xgboost_order": round(prec(alerts["our_score"].sort_values(ascending=False).index), 4),
                 "random_order": round(y.mean(), 4)})
qm = pd.DataFrame(rows); qm.to_csv(OUT / "queue_metrics.csv", index=False)

# signal lift on training months (fraud rate with signal / overall)
sig = {"device_fraud_count>0": train["device_fraud_count"] > 0, "name_email_similarity<0.2": train["name_email_similarity"] < 0.2,
       "phone_home_valid=0": train["phone_home_valid"] == 0, "keep_alive_session=0": train["keep_alive_session"] == 0,
       "email_is_free=1": train["email_is_free"] == 1, "foreign_request=1": train["foreign_request"] == 1,
       "device_distinct_emails_8w>1": train["device_distinct_emails_8w"] > 1, "prev_address_months_count=-1": train["prev_address_months_count"] == -1,
       "housing_status=BA": train["housing_status"] == "BA", "date_of_birth_distinct_emails_4w<=3": train["date_of_birth_distinct_emails_4w"] <= 3}
base_rate = train["fraud_bool"].mean()
pd.DataFrame([{"signal": k, "share_of_apps": round(m.mean(), 4), "fraud_rate": round(train.loc[m, "fraud_bool"].mean(), 4),
               "lift_vs_overall": round(train.loc[m, "fraud_bool"].mean() / base_rate, 2)} for k, m in sig.items()]
             ).sort_values("lift_vs_overall", ascending=False).to_csv(OUT / "signal_lift.csv", index=False)

keep = ["case_id", "month", "fraud_bool", "model_score", "our_score", "analyst_fraud_votes", "reasons"] + feats
alerts[keep].to_parquet(OUT / "alerts_scored.parquet", index=False)

auc = roc_auc_score(y, alerts["our_score"]); ap = average_precision_score(y, alerts["our_score"])
auc0 = roc_auc_score(y, alerts["model_score"])
card = f"""# Model card: alert-queue re-ranker (portfolio project)

- Training: {len(train):,} applications from months {sorted(train['month'].unique().tolist())} (fraud rate {base_rate:.2%}); alerts come from months {alert_months} and were never used for training.
- Alerts: {len(alerts):,} cases, fraud rate {y.mean():.2%}.
- Within-alert AUC: our XGBoost {auc:.3f} vs provided alert-model score {auc0:.3f}; average precision {ap:.3f}.
- Synthetic analysts (50): mean accuracy {np.mean(exp_acc):.1%} (range {min(exp_acc):.1%}-{max(exp_acc):.1%}); mean fraud catch rate {np.mean(exp_tpr):.1%}.
- Reason codes: top-3 positive TreeSHAP contributions per case, translated to plain-language phrases.
- Limits: synthetic data and synthetic analysts; feature meanings follow the BAF datasheet; not for real decisions.
"""
(OUT / "model_card.md").write_text(card)
print(card); print(qm.to_string(index=False))
print(pd.read_csv(OUT / "signal_lift.csv").to_string(index=False))
print(alerts.loc[alerts["our_score"].idxmax(), ["case_id", "fraud_bool", "our_score", "reasons"]].to_dict())
