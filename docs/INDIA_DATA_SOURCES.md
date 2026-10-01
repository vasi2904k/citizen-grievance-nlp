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
**38.9% accuracy and 42.1% six-class macro-F1**. Results differ sharply by language:
English is approximately **63.0% accuracy**, while Hindi is approximately
**14.8%**. These figures are diagnostic only because the source is TTS-generated,
the review is assistant-assisted rather than independent human validation, and
the source covers only six of the fourteen project departments.

For a fair six-class comparison, run:

```bash
python scripts/train_gcd_auxiliary_model.py
python scripts/evaluate_gcd_systems.py
```

On the untouched 27-row GCD test split, the current comparison is:

| System | Overall accuracy | Six-class macro-F1 | English accuracy | Hindi accuracy |
|---|---:|---:|---:|---:|
| Current 14-class model, restricted to six classes | 40.7% | 44.6% | 57.1% | 23.1% |
| Six-class auxiliary model | 48.1% | 44.8% | 21.4% | 76.9% |
| Rules plus auxiliary model | 85.2% | 83.6% | 78.6% | 92.3% |

This is a small, assistant-reviewed, TTS-generated test split grouped by source
file rather than a human-reviewed benchmark. The hybrid result should be
treated as a diagnostic of high-signal vocabulary coverage, not as production
performance.

## Independent review benchmark

The independent-review workflow is prepared by:

```bash
python scripts/prepare_independent_hindi_benchmark.py
```

This creates:

- `india_gcd_second_review_queue.csv`: all 108 previously approved rows plus
  the 12 generic rows, with blank independent-review fields.
- `india_hindi_benchmark_candidates.csv`: the GCD rows plus additional
  English, Devanagari Hindi, and Hinglish candidate variations for all six
  departments.

The candidate benchmark has 56 rows per department in the current local
generation (22 English, 22 Hindi, and 12 Hinglish), and all rows are marked
`pending_independent_review`. The independent reviewer must populate
`independent_review_status=independently_approved` and `final_department`
before they count toward the 50–100-row-per-department target. The generated
candidate rows are not presented as naturally collected complaints or human
labels.

After review, run:

```bash
python scripts/evaluate_independent_benchmark.py
```

The evaluator reports six-class macro-F1, per-class precision/recall,
confusion matrices, and English/Hindi/Hinglish metrics for the current
14-class model, six-class auxiliary model, and hybrid system. Training and
testing are grouped by source/template to prevent leakage.

Production integration is intentionally deferred. Until the independent
benchmark reaches the review and sample-size threshold, the current 14-class
model remains the default and the auxiliary/hybrid systems remain diagnostic
only.

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
