"""Step 2: AI triage assistant for fraud investigators (runs fully local via Ollama).

For each flagged account-opening application, a local LLM drafts an investigator-facing case summary
from the model score and the reason codes produced in step 1. The LLM NEVER makes the decision:
a human approves or overrides, and every step is logged for audit.

Guardrails
  * JSON-only output validated against a fixed schema (one retry, then a deterministic fallback).
  * Grounding check: every risk signal the LLM cites must come from the case's own reason codes,
    so it cannot invent facts. Ungrounded output is rejected.
  * Suggested checks must come from an approved list.
  * Consistency check: if the LLM's priority disagrees with the score band, the case is flagged
    for senior review instead of being silently accepted.
  * Audit log (JSONL): input packet, prompt hash, model, raw output, validation results, final
    status, and the human decision (simulated here with one synthetic analyst's call).

Usage: python prep/triage_assistant.py [n_top] [n_random]
Outputs (output/): triage_audit_log.jsonl, triage_results.parquet, triage_eval.md, sample_cases.md
"""
import json, hashlib, time, sys, pathlib, datetime, urllib.request, re
import pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
RAW = ROOT / "data_raw" / "FiFAR"
MODEL = "qwen2.5:7b"
OLLAMA = "http://127.0.0.1:11434/api/chat"
N_TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 150
N_RAND = int(sys.argv[2]) if len(sys.argv) > 2 else 100

CHECKS = ["Verify identity document", "Confirm email ownership", "Verify phone number",
          "Review device history", "Verify address", "Check for linked applications",
          "Contact applicant", "Approve with standard monitoring"]
SYSTEM = f"""You assist a bank fraud investigator reviewing a flagged account-opening application.
You do NOT decide whether the application is fraud and you do NOT set its priority; the risk model ranks
cases and a human investigator decides. Use ONLY the facts provided. Field meanings are partly anonymized,
so never describe a value as high, low, good, or bad; state the values as given.
Return a single JSON object with exactly these keys:
  "investigator_note": string, at most 35 words: what to verify first and which flags it addresses
  "risk_signals": list of 1-4 strings, each copied exactly from the provided reason codes
  "suggested_checks": list of 1-3 strings chosen only from: {json.dumps(CHECKS)}
Use "Approve with standard monitoring" only by itself, never together with investigation checks.
Match checks to flags: device or multiple-email flags -> "Review device history"; name/email mismatch ->
"Confirm email ownership"; phone not valid -> "Verify phone number"; address flags -> "Verify address"; velocity or
same-ZIP volume -> "Check for linked applications"; anything else -> "Verify identity document".
Refer to flags using the reason-code wording, never raw field names.
Example note format (fill in from THIS case only): "Start with <check>: <flag copied from the reason codes>.
Then <check>, since <another flag copied from the reason codes>."
"""

def band(s): return "HIGH" if s >= 0.6 else ("MEDIUM" if s >= 0.3 else "LOW")

def call(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "format": "json", "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["message"]["content"]

JUDGMENT = re.compile(r"\b(?:low|high|poor|weak|strong|good|bad|large|small)\s+(?:credit|income|limit|balance|deposit)\b|"
                      r"\b(?:credit(?: risk)? score|income|limit|credit limit|balance)\s+(?:is\s+)?(?:low|high|poor|weak)\b", re.I)

def validate(out, reasons):
    errs = []
    if set(out) != {"investigator_note", "risk_signals", "suggested_checks"}: errs.append(f"keys {sorted(out)}")
    note = out.get("investigator_note")
    if not isinstance(note, str) or not note.strip() or len(note.split()) > 35: errs.append("note length")
    sig = out.get("risk_signals") or []
    if not (isinstance(sig, list) and 1 <= len(sig) <= 4): errs.append("risk_signals count")
    texts = [r["text"].lower() for r in reasons]
    ungrounded = [x for x in sig if not any(str(x).lower().strip(" .") in t or t in str(x).lower() for t in texts)]
    if ungrounded: errs.append(f"ungrounded signals {ungrounded}")
    ch = out.get("suggested_checks") or []
    if not (isinstance(ch, list) and 1 <= len(ch) <= 3 and all(c in CHECKS for c in ch)): errs.append("checks")
    if "Approve with standard monitoring" in ch and len(ch) > 1: errs.append("contradictory checks (approve + investigate)")
    # numeric grounding: every number in the note must appear in this case's reason codes
    src = " ".join(texts)
    bad_nums = [n for n in re.findall(r"\d+(?:\.\d+)?", note or "") if n not in src]
    if bad_nums: errs.append(f"numbers not in case facts {bad_nums}")
    judg = JUDGMENT.findall(note or "")
    if judg: errs.append(f"unsupported value judgment {judg}")
    return errs

alerts = pd.read_parquet(OUT / "alerts_scored.parquet")
experts = pd.read_parquet(RAW / "synthetic_experts" / "expert_predictions.parquet")
experts = experts.reindex(alerts["case_id"].values).reset_index(drop=True) if experts.index.name == "case_id" else experts
alerts = alerts.reset_index(drop=True)
alerts["queue_rank"] = alerts["model_score"].rank(ascending=False, method="first").astype(int)
top = alerts.nsmallest(N_TOP, "queue_rank")
rnd = alerts.drop(top.index).sample(N_RAND, random_state=11)
cases = pd.concat([top.assign(sample="top_of_queue"), rnd.assign(sample="random")])
reviewer = experts.columns[0]   # one synthetic analyst plays the human reviewer

log_f = open(OUT / "triage_audit_log.jsonl", "w")
rows = []
for _, c in cases.iterrows():
    reasons = json.loads(c["reasons"])
    packet = {"case_id": int(c["case_id"]), "risk_score": round(float(c["model_score"]), 3), "queue_rank": int(c["queue_rank"]),
              "reason_codes": [r["text"] for r in reasons]}   # the LLM sees only score, rank and reason codes
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(packet)}]
    prompt_hash = hashlib.sha256(json.dumps(msgs).encode()).hexdigest()[:16]
    attempts, out, errs, raw, t0 = 0, None, ["not run"], "", time.time()
    while attempts < 2 and errs:
        attempts += 1
        try:
            raw = call(msgs); out = json.loads(raw); errs = validate(out, reasons)
        except Exception as e:
            errs = [f"exception {type(e).__name__}"]
        if errs and attempts == 1:
            msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": f"Invalid: {errs}. Return corrected JSON only, using only the provided reason codes."}]
    latency = round(time.time() - t0, 2)
    priority = band(packet["risk_score"])           # set by the risk model, never by the LLM
    facts = f"Score {packet['risk_score']}, rank {packet['queue_rank']} in the queue. Flags: " + "; ".join(packet["reason_codes"]) + "."
    if errs:   # deterministic fallback: never ship an unvalidated AI note
        status = "FALLBACK_TEMPLATE"
        out = {"investigator_note": "Assistant note unavailable; review the flags above.",
               "risk_signals": packet["reason_codes"][:3], "suggested_checks": ["Verify identity document"]}
    elif priority == "HIGH" and out["suggested_checks"] == ["Approve with standard monitoring"]:
        status = "ASSISTANT_SUGGESTS_APPROVAL_ON_HIGH_SCORE_SENIOR_REVIEW"
    else:
        status = "ACCEPTED_FOR_HUMAN_REVIEW"
    out = {**out, "priority": priority, "summary": facts + " " + out["investigator_note"]}
    human = int(experts.loc[_, reviewer])
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "model": MODEL, "prompt_hash": prompt_hash,
           "input": packet, "raw_output": raw, "attempts": attempts, "validation_errors": errs, "status": status,
           "assistant_output": out, "latency_s": latency,
           "human_decision": {"reviewer": reviewer, "decision": "FRAUD" if human else "LEGITIMATE", "note": "synthetic analyst stands in for the human"},
           "ground_truth": int(c["fraud_bool"]), "sample": c["sample"]}
    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
    rows.append({"case_id": packet["case_id"], "sample": c["sample"], "score": packet["risk_score"], "priority": priority,
                 "status": status, "attempts": attempts, "latency_s": latency, "human_fraud": human, "fraud": int(c["fraud_bool"])})
log_f.close()

r = pd.DataFrame(rows); r.to_parquet(OUT / "triage_results.parquet", index=False)
acc = r[r.status != "FALLBACK_TEMPLATE"]
lines = [f"# Triage assistant evaluation ({MODEL}, local via Ollama)", "",
         f"- Cases: {len(r)} ({(r['sample']=='top_of_queue').sum()} from the top of the queue, {(r['sample']=='random').sum()} random alerts)",
         f"- Valid on first try: {(r.attempts==1).mean():.0%}; valid after one retry: {(r.status!='FALLBACK_TEMPLATE').mean():.0%}; fallback template used: {(r.status=='FALLBACK_TEMPLATE').mean():.0%}",
         f"- Ungrounded (invented) signals shipped to investigators: 0 by construction (rejected by the grounding check)",
         f"- Assistant suggested plain approval on a HIGH-score case (sent to senior review): {(r.status=='ASSISTANT_SUGGESTS_APPROVAL_ON_HIGH_SCORE_SENIOR_REVIEW').mean():.0%}",
         f"- Median latency per case: {r.latency_s.median():.1f}s on an Apple M4 Mac mini", "",
         "| Priority (from risk score) | Cases | Actual fraud rate |", "|---|---|---|"]
for p in ("HIGH", "MEDIUM", "LOW"):
    s = r[r.priority == p]
    if len(s): lines.append(f"| {p} | {len(s)} | {s.fraud.mean():.0%} |")
lines += ["", f"Reviewer ({reviewer}, synthetic) accuracy on these cases: {(r.human_fraud == r.fraud).mean():.0%}.",
          "", "Limits: synthetic applications and synthetic analysts; the assistant explains cases; the risk model sets priority and a human decides."]
(OUT / "triage_eval.md").write_text("\n".join(lines))

ex = []
for rec in map(json.loads, open(OUT / "triage_audit_log.jsonl")):
    if rec["status"] != "FALLBACK_TEMPLATE" and len(ex) < 8:
        o = rec["assistant_output"]
        ex.append(f"### Case {rec['input']['case_id']} (score {rec['input']['risk_score']}, rank {rec['input']['queue_rank']})\n"
                  f"**Priority:** {o['priority']}  \n**Summary:** {o['summary']}  \n"
                  f"**Signals:** {'; '.join(o['risk_signals'])}  \n**Checks:** {'; '.join(o['suggested_checks'])}  \n"
                  f"**Status:** {rec['status']} | **Human decision:** {rec['human_decision']['decision']} | **Actual:** {'fraud' if rec['ground_truth'] else 'legitimate'}\n")
(OUT / "sample_cases.md").write_text("# Sample triage cases\n\n" + "\n".join(ex))
print("\n".join(lines))
