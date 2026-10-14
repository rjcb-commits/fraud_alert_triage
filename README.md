# Fraud Alert Triage: Risk Ranking + a Guardrailed Local AI Assistant

This project looks at **bank account-opening fraud**: rank a queue of fraud alerts by risk, explain each alert in plain English, and use a **local LLM** to draft investigator notes, with checks in code that tie every flag and number in a note to the case data, an independent audit of the notes, and a full audit log. The AI explains; the risk model sets priority; a human decides.

> Synthetic data only (Feedzai BAF / FiFAR). Built to demonstrate fraud-operations analytics and safe AI workflow design, not for real decisions.

**Interactive dashboard (Tableau Public):** [Fraud Alert Triage: Ranking, Reviewers & Guardrailed AI](https://public.tableau.com/app/profile/ray.jack/viz/FraudAlertTriage/1-Queue)

![Same alerts, 50 reviewers](images/dashboard-reviewers.png)

| Queue | AI triage |
|---|---|
| ![Queue](images/dashboard-queue.png) | ![AI triage](images/dashboard-ai-triage.png) |

## Key results

**1. Ranking the queue matters.** Of 30,622 alerts (12.1% fraud), an investigator working the top 100 by risk score finds **74% fraud**, about **6x** random order (12%).

| Daily review capacity | Fraud rate, risk-ranked | Fraud rate, random order |
|---|---|---|
| 100 | 74% | 12% |
| 500 | 53% | 12% |
| 2,000 | 37% | 12% |

But **two-thirds of the fraud sits in low-score alerts** (score < 0.2, where only 7-13% of cases are fraud), so "only work high scores" misses most of it. That is the capacity vs. coverage trade-off fraud teams manage.

**2. Reviewers vary widely.** Across 50 synthetic analysts, accuracy ranges from **33% to 96%**. One catches 99% of fraud but flags 76% of legitimate applicants; another is 96% accurate, catches 79%, and flags only 1.5%. Consistent case notes are one way to narrow that spread.

**3. The AI triage assistant (local Qwen 2.5 7B via Ollama, Apple M4):** on 250 cases, **98% of AI notes passed every guardrail on the first try** and 100% after one retry, **0 notes cited a flag or number the case doesn't have** (confirmed by an independent audit), no fallbacks to the deterministic template, median **3.7 s per case**.

| Priority (from risk score) | Cases | Actual fraud rate |
|---|---|---|
| HIGH | 150 | 71% |
| MEDIUM | 9 | 33% |
| LOW | 91 | 12% |

The audit (`prep/audit_notes.py`) re-reads every note and checks its flags and numbers against the case's reason codes: 0 of 250 in Run 4 ([`output/note_audit.md`](output/note_audit.md)). Run against Run 3, the same audit finds 19 of 244 accepted notes citing "same-ZIP volume", a phrase the model copied from the prompt's own rule list ([`output/note_audit_run3.md`](output/note_audit_run3.md)).

Example note (case 559428, random sample, score 0.05, legitimate):
> Score 0.054, rank 29006 in the queue. Flags: name and email don't match (similarity 0.09); 0.00 days since request; date of birth seen with 3 emails in 4 weeks. *First, confirm email ownership: name and email don't match (similarity 0.09). Then verify identity document: date of birth seen with 3 emails in 4 weeks.*

More in [`output/sample_cases.md`](output/sample_cases.md).

## How the guardrails evolved (four runs)

| Run | Design | Valid first try | Fallback |
|---|---|---|---|
| 1 | LLM writes summary + sets priority; prompt says "don't invent" | 89% | 2% |
| 2 | Added code checks for unsupported judgments ("low credit score") | 57% | 40% |
| 3 | LLM only sees score, rank and reason codes; facts filled by code; LLM writes a short note; every number must exist in the case | 98% | 2% |
| 4 | Every flag and number in the note text must match the case's reason codes | **98%** | **0%** |

Run 1 *looked* fine but most summaries called values "low" or "high", which the anonymized data can't support. Run 2's stricter checks caught that, and also showed the model copying numbers from the prompt's example into real cases. The fix was architectural: **narrow what the LLM is allowed to do and enforce the rules in code instead of trusting the prompt.**

Run 3 checked the structured signal list but not the flags written into the note itself, and its number check only looked for digits anywhere in the case. Auditing the notes afterwards found 19 of 244 citing "same-ZIP volume", copied from the prompt's rule list. Run 4 checks the note text too, matches numbers as whole numbers, and tells the model the rule list names kinds of flags, not facts about the case. In Run 4, 5 of 250 notes needed a retry and all 5 passed; in Run 3, all 6 retries had failed and fallen back to the template.

## Guardrails

- **Fixed JSON schema**, one retry, then a deterministic fallback; an unvalidated AI note is never shown.
- **Grounding** (`prep/note_checks.py`): every signal in the `risk_signals` list and every flag written into the note must match the case's own reason codes, and every number in the note must appear as a whole number in them.
- **No value judgments** on anonymized fields (high/low/good/bad), enforced by pattern checks.
- **Approved actions only**; "approve" cannot be combined with investigation steps; an AI approval suggestion on a HIGH-score case is routed to senior review.
- **Priority comes from the risk model, never the LLM.**
- **Audit log** (`output/triage_audit_log.jsonl`): input, prompt hash, model, raw output, validation errors, status, and the human decision for every case.
- **Runs fully local**: no application data leaves the machine.

## Method

1. `prep/score_alerts.py`: XGBoost trained on months 0-2 (397k applications, never the alert months); scores the 30,622 alerts; per-case **reason codes** from TreeSHAP contributions translated to plain language; queue precision at fixed capacity; signal lift.
2. `prep/triage_assistant.py`: the guardrailed LLM assistant (Ollama, `qwen2.5:7b`, temperature 0). A synthetic analyst stands in for the human reviewer.
3. `prep/build_tableau_tables.py`: tables for the Tableau dashboard (`tableau/`).
4. `prep/audit_notes.py`: re-reads the audit log and checks each accepted note's flags and numbers against the case's reason codes.

My XGBoost re-ranker (within-alert AUC 0.659) did not beat the dataset's own alert-model score (0.677), so the queue ranks by the provided score and uses my model only for explanations.

## Limitations

- The checks confirm each flag and number in a note comes from the case. They don't confirm the note pairs the right check with each flag.
- Synthetic applications and synthetic analysts; some fields are anonymized codes (e.g. housing status "BA"), so several reasons read as codes.
- The 7B model's notes are formulaic and sometimes pair a check with a flag imprecisely; a larger model or a deterministic flag-to-check map would improve this.
- Evaluated on 250 cases (150 top-of-queue + 100 random), not the full alert set.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download FiFAR (CC BY): https://springernature.figshare.com/articles/dataset/Financial_Fraud_Alert_Review_Dataset/28351172  -> unzip into data_raw/
python prep/score_alerts.py
brew install ollama && ollama serve &   # then: ollama pull qwen2.5:7b
python prep/triage_assistant.py 150 100
python prep/build_tableau_tables.py
python prep/audit_notes.py
```

## Data credit

FiFAR (Financial Fraud Alert Review) dataset (figshare, CC BY license), built on the Bank Account Fraud (BAF) suite by Feedzai (Jesus et al., "Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation", NeurIPS 2022). This repository is not affiliated with Feedzai.
