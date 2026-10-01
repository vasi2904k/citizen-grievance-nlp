# India grievance data sources

## Downloaded local source

The local working copy includes the MIT-licensed **GCD Government Complaints
Dataset** from Hugging Face:

- Source: <https://huggingface.co/datasets/Munavvara-17/GCD-Government_Compliants_Dataset>
- Local directory: `data/raw/india_government_complaints_gcd/`
- Contents: 120 Hindi and English TTS-generated complaint audio files,
  `metadata.csv`, the source dataset README, and `SOURCE_INFO.json`.
- Categories: electricity, healthcare, ration/Aadhaar, roads/transport,
  sanitation, and water supply.

The dataset is intentionally ignored by Git because it is downloaded input data,
not application source. `metadata.csv` contains the supplied transcripts and
category labels. The audio and transcripts must be independently reviewed before
being added to a training or validation split.

The derived review queue is generated locally at
`data/evaluation/india_gcd_review_queue.csv`:

```text
grievance_text, source_category, language, source_file,
review_status, mapped_department
```

Every row is currently marked `synthetic_tts_pending_manual_review`. The
category-to-department mapping is a provisional review suggestion, not a ground
truth label. The queue is kept separate from
`data/evaluation/india_department_examples.csv` and is not used for training.

The assistant review output is generated at
`data/evaluation/india_gcd_reviewed.csv`. It contains 108
`assistant_reviewed_approved` rows and 12
`assistant_reviewed_needs_second_review` rows. The latter are generic
transcripts whose source category is plausible but whose text does not contain
enough service-specific evidence for safe standalone routing evaluation.
Independent human review is still required before any rows are promoted to a
training or benchmark split.

## Baseline result before merging

Run:

```bash
python scripts/prepare_gcd_derived_dataset.py
python scripts/evaluate_gcd_baseline.py
```

On the 108 assistant-approved rows, the current model baseline is approximately
**38.9% accuracy and 21.1% macro-F1**. Results differ sharply by language:
English is approximately **63.0% accuracy**, while Hindi is approximately
**14.8%**. These figures are diagnostic only because the source is TTS-generated,
the review is assistant-assisted rather than independent human validation, and
the source covers only six of the fourteen project departments.

## Reproduction

The source dataset can be downloaded again from its Hugging Face dataset page.
The current local copy was downloaded from the repository's `main` revision on
2026-10-02. Check `SOURCE_INFO.json` for the recorded source and download
counts.

## Usage restrictions and quality notes

- Keep the source license and attribution with any derived dataset.
- Do not represent the TTS-generated complaints as naturally collected citizen
  reports.
- Do not use the data as a production ground-truth benchmark without human
  review.
- Map the source categories to the project's India departments only through an
  explicit, reviewable mapping.
- Preserve language and source fields so Hindi/English performance can be
  evaluated separately.
