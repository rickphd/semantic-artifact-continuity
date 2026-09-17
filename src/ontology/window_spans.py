"""A0-F1: map lexical match spans to the stage's existing token spans.

The matched expression remains inside the scoring window. Filtering and
punctuation cleanup occur afterwards in the unchanged stage-specific code.
No alias deduplication, detection, polarity rule or tokenizer is changed here.
"""
import re


def context_bounds(spans, match, radius=5, whole_expression=True):
    lexical = list(re.finditer(r"\w", match.group()))
    if not lexical:
        raise ValueError("Concept match has no lexical characters")
    start = match.start() + lexical[0].start()
    end = match.start() + lexical[-1].end()
    covered = [i for i, token in enumerate(spans)
               if token.start() < end and token.end() > start]
    if not covered:
        raise ValueError("Concept match does not overlap any token")
    first, last = covered[0], covered[-1]
    right_anchor = last if whole_expression else first
    return max(0, first - radius), min(len(spans), right_anchor + radius + 1)
