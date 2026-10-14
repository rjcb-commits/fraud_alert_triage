"""Checks on the investigator note text, shared by triage_assistant.py (validation) and audit_notes.py (audit).

Notes follow the format "First, <check>: <flag>. Then <check>: <flag>." Every flag written into the
note must be one of the case's own reason codes, and every number must appear as a whole number in them.
"""
import re

SENTENCE = re.compile(r"(?<=\.)\s+(?=[A-Z])")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def note_flags(note):
    """Flags cited in the note: the text after the colon (or 'since') in each sentence."""
    flags = []
    for sentence in SENTENCE.split(note.strip()):
        sentence = sentence.strip().rstrip(".")
        if ": " in sentence:
            tail = sentence.split(": ", 1)[1]
        elif " since " in sentence:
            tail = sentence.split(" since ", 1)[1]
        else:
            continue
        flags += [f.strip() for f in re.split(r";\s+|,\s+", tail) if f.strip()]
    return flags


def _norm(text):
    """Lower-case and drop plural 's' so '1 email' and '1 emails' compare equal."""
    return re.sub(r"(?<=[a-z])s\b", "", text.lower())


def unsupported_flags(note, reason_codes):
    """Flags in the note that do not match any of the case's reason codes."""
    codes = [_norm(c) for c in reason_codes]
    out = []
    for f in note_flags(note):
        fl = _norm(f)
        if not any(fl == c or (len(fl) >= 6 and fl in c) or c in fl for c in codes):
            out.append(f)
    return out


def unsupported_numbers(note, reason_codes):
    """Numbers in the note that do not appear as a whole number in the case's reason codes."""
    known = set(NUMBER.findall(" ".join(reason_codes)))
    return [n for n in NUMBER.findall(note) if n.rstrip(",") not in known]


def tidy_note(note, checks):
    """Lower-case an approved check name when it follows 'First, ', 'Start with ' or 'Then '."""
    for c in checks:
        lower = c[0].lower() + c[1:]
        for lead in ("First, ", "Start with ", "Then "):
            note = note.replace(lead + c, lead + lower)
    return note
