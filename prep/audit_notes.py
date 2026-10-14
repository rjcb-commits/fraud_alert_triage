"""Step 4: independent audit of the accepted AI notes in the latest audit log.

Re-reads output/triage_audit_log.jsonl and checks every accepted note's text against the case's own
reason codes with the same rules the validator uses (prep/note_checks.py): flags in the note must be
reason codes, and numbers must appear as whole numbers in them.

Usage: python prep/audit_notes.py [audit_log.jsonl]
Output: output/note_audit.md
"""
import collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from note_checks import unsupported_flags, unsupported_numbers

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
LOG = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "triage_audit_log.jsonl"
rows = [json.loads(line) for line in open(LOG)]
accepted = [r for r in rows if r["status"] != "FALLBACK_TEMPLATE"]

unsupported = collections.Counter()
cases = []
for r in accepted:
    codes = r["input"]["reason_codes"]
    note = r["assistant_output"]["investigator_note"]
    missing = unsupported_flags(note, codes) + unsupported_numbers(note, codes)
    if missing:
        unsupported.update(missing)
        cases.append((r["input"]["case_id"], missing, r["input"]["reason_codes"]))

lines = [
    "# Note audit: flags or numbers in the note text that are not in the case's reason codes",
    "",
    f"- Audit log: {LOG.name}",
    f"- Accepted notes: {len(accepted)}",
    f"- Notes citing a flag or number the case does not have: {len(cases)} ({len(cases) / len(accepted):.1%})",
    "",
    "| Flag or number in note | Notes |",
    "|---|---|",
    *[f"| {flag} | {n} |" for flag, n in unsupported.most_common()],
    "",
    "| Case | Unsupported text | Case reason codes |",
    "|---|---|---|",
    *[f"| {cid} | {'; '.join(m)} | {'; '.join(codes)} |" for cid, m, codes in cases],
    "",
]
(OUT / "note_audit.md").write_text("\n".join(lines))
print(f"{LOG.name}: {len(cases)} of {len(accepted)} accepted notes cite a flag or number not in the case's reason codes")
