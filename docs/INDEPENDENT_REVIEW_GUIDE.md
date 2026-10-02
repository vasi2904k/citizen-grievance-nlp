# Independent benchmark review guide

Review this file:

```text
data/evaluation/india_hindi_benchmark_candidates.csv
```

The file contains 336 candidate rows:

- 132 English
- 132 Devanagari Hindi
- 72 Hinglish/Roman Hindi
- 56 rows for each of the six departments

## Fields to complete

For every row, fill these fields:

| Field | Allowed value |
|---|---|
| `independent_review_status` | `independently_approved` or `independently_excluded` |
| `final_department` | One of the six departments below, or blank if excluded |
| `independent_reviewer` | Your reviewer name or initials |
| `independent_review_notes` | Short reason for the decision |

Allowed departments:

- `Water Supply & Sewerage`
- `Electricity & Power`
- `Roads & Transport`
- `Public Health`
- `Environment & Pollution`
- `Public Distribution System`

Do not change `grievance_text`, `language`, `source_group`, or `template_id`.
Those fields are used to preserve provenance and prevent train/test leakage.

## Review decision rules

Approve a row only when the complaint text contains enough evidence for one
specific department. Use the department suggested by `mapped_department` only
when the text supports it; do not treat that column as ground truth.

Exclude a row when it is:

- too generic to identify a department;
- ambiguous between two departments;
- not a complaint;
- unintelligible or incorrectly transcribed;
- missing enough context to make a reliable decision.

For an approved row, set `final_department` to the reviewed department. For an
excluded row, leave `final_department` blank.

## No-response escalation flag

The 12 generic rows listed below are flagged with:

- `escalation_flag=true`
- `escalation_intent=no_response_escalation`

This means the complaint expresses a Voice-of-the-Citizen escalation signal:
the citizen reports that a prior complaint was ignored, delayed, unresolved,
or blocked by an unresponsive office, staff member, system, or helpline.
This flag is independent of the department label. A reviewer may assign a
department if the text supports one, or exclude the row if it is too generic.

## The 12 generic GCD rows

These rows need special attention because the original assistant review held
them as generic:

| Text | Suggested source category | Language |
|---|---|---|
| Complaints are raised but no action is taken. | electricity | English |
| We had to wait for hours, and no one responded. | healthcare | English |
| Staff is uncooperative and careless. | healthcare | English |
| The system is always down when we visit the office. | ration_aadhar | English |
| The helpline number is always busy or unreachable. | ration_aadhar | English |
| Despite repeated complaints, the issue is not resolved. | water_supply | English |
| कई बार शिकायत की, लेकिन कोई कार्रवाई नहीं होती। | electricity | Hindi |
| स्टाफ असहयोगी और लापरवाह है। | healthcare | Hindi |
| लैब जांच की रिपोर्ट बिना कारण देर से मिलती है। | healthcare | Hindi |
| जब भी हम कार्यालय जाते हैं, सिस्टम डाउन रहता है। | ration_aadhar | Hindi |
| हेल्पलाइन नंबर हमेशा व्यस्त या अप्राप्य रहता है। | ration_aadhar | Hindi |
| कई बार शिकायत की, लेकिन कोई समाधान नहीं हुआ। | water_supply | Hindi |

For these 12 rows, it is acceptable and often safer to use
`independently_excluded` with a blank `final_department` unless you can infer
the department confidently from the text.

## Run the independent evaluation

After completing the CSV, verify that:

```powershell
Import-Csv data\evaluation\india_hindi_benchmark_candidates.csv |
  Group-Object independent_review_status
```

Then run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_independent_benchmark.py
```

The evaluator will use only rows marked `independently_approved` with a valid
`final_department`. It reports accuracy, six-class macro-F1, per-class
precision/recall, confusion matrices, and separate English, Hindi, and
Hinglish results for all three systems.

Do not approve production integration based on the current assistant-only
diagnostics. Use the independent results to decide whether the auxiliary Hindi
path improves consistently over the current 14-class model.
