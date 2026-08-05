# Data Notice

## Scope

`data/gold/gold_enriched_ontology.parquet` contains 1,614 public Reddit posts,
manual sentiment labels, 37 ontology-derived variables, and the fixed
train/validation/test split used by the released experiments.

A version-specific copy of the Gold corpus is available from Hugging Face at
https://doi.org/10.57967/hf/9852.

## Minimization

The release omits Reddit account names, full-name fields, URLs, voting and
engagement metadata, media payloads, flair metadata, and the raw API payload.
Corpus authors are represented by local ordinal identifiers. The mapping from
Reddit account names to those identifiers is not distributed.

Post IDs and text are retained because they are required to inspect the source
records and reproduce the text-based experiments. They may permit a reader to
locate the original public post. Do not use the dataset to profile, contact, or
attempt to identify individual Reddit users.

## Rights And Reuse

The sentiment labels, fixed split assignments, ontology-derived variables,
documentation, and release metadata created by the dataset authors are
licensed under the Creative Commons Attribution 4.0 International license:

https://creativecommons.org/licenses/by/4.0/

The MIT software license does not apply to Reddit-authored text. Original post
content remains subject to the rights of its authors and applicable platform
terms. The repository does not transfer ownership of third-party content, and
CC BY 4.0 does not apply to that content. Users are responsible for complying
with applicable laws, research-ethics requirements, and Reddit terms.

## Removal Requests

Content-removal or data-governance requests may be sent to the corresponding
author at `rflores@usfq.edu.ec`.
