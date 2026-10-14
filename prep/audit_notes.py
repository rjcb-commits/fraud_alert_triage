"""Step 4: audit accepted AI notes for flags the validator does not check.

The validator in triage_assistant.py checks the structured risk_signals list and the numbers in each
note. This script reads the audit log and checks the flags written into the note text itself
("Start with <check>: <flag>. Then <check>: <flag>.") against the case's own reason codes.

Usage: python prep/audit_notes.py
Output: output/note_audit.md
"""
import collections, json, pathlib, re

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
FLAG = re.compile(r"(?::|since)\s*([^.]+?)(?:\.|$)")

rows = [json.loads(line) for line in open(OUT / "triage_audit_log.jsonl")]
accepted = [r for r in rows if r["status"] != "FALLBACK_TEMPLATE"]

unsupported = collections.Counter()
cases = []
for r in accepted:
    codes = [c.lower() for c in r["input"]["reason_codes"]]
    note = r["assistant_output"]["investigator_note"]
    flags = [f.strip().rstrip(".") for f in FLAG.findall(note)]
    missing = [f for f in flags if not any(f.lower() in c or c in f.lower() for c in codes)]
    if missing:
        unsupported.update(missing)
        cases.append((r["input"]["case_id"], missing, r["input"]["reason_codes"]))

lines = [
    "# Note audit: flags in the note text that are not in the case's reason codes",
    "",
    f"- Accepted notes: {len(accepted)}",
    f"- Notes citing at least one flag the case does not have: {len(cases)} ({len(cases) / len(accepted):.1%})",
    "",
    "| Flag in note | Notes |",
    "|---|---|",
    *[f"| {flag} | {n} |" for flag, n in unsupported.most_common()],
    "",
    "| Case | Unsupported flag(s) | Case reason codes |",
    "|---|---|---|",
    *[f"| {cid} | {'; '.join(m)} | {'; '.join(codes)} |" for cid, m, codes in cases],
    "",
]
(OUT / "note_audit.md").write_text("\n".join(lines))
print(f"{len(cases)} of {len(accepted)} accepted notes cite a flag not in the case's reason codes")
