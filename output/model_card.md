# Model card: alert-queue re-ranker (portfolio project)

- Training: 397,039 applications from months [0, 1, 2] (fraud rate 0.98%); alerts come from months [3, 4, 5, 6, 7] and were never used for training.
- Alerts: 30,622 cases, fraud rate 12.13%.
- Within-alert AUC: our XGBoost 0.659 vs provided alert-model score 0.677; average precision 0.250.
- Synthetic analysts (50): mean accuracy 70.7% (range 33.1%-96.1%); mean fraud catch rate 84.3%.
- Reason codes: top-3 positive TreeSHAP contributions per case, translated to plain-language phrases.
- Limits: synthetic data and synthetic analysts; feature meanings follow the BAF datasheet; not for real decisions.
