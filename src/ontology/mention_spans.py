"""Deterministic per-concept lexical occurrence selection for A0-F2.

Exact lexical spans are duplicates. Optional longest-first exclusion handles
overlapping expressions. Adjacent spans and spans for different concepts remain
independent. Windows may overlap even when occurrences do not.
"""
import re


def lexical_span(match):
    letters = list(re.finditer(r"\w", match.group()))
    if not letters:
        raise ValueError("Matched expression has no lexical characters")
    return match.start() + letters[0].start(), match.start() + letters[-1].end()


def select_mentions(text, patterns, policy="longest"):
    if policy not in {"exact", "longest"}:
        raise ValueError(policy)
    unique = {}
    for pattern in sorted(set(patterns)):
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = lexical_span(match)
            unique.setdefault(span, match)
    ranked = sorted(unique, key=lambda span: (-(span[1] - span[0]), span[0], span[1]))
    retained = []
    for span in ranked:
        if policy == "exact" or not any(span[0] < end and start < span[1]
                                        for start, end in retained):
            retained.append(span)
    return [unique[span] for span in sorted(retained)]
