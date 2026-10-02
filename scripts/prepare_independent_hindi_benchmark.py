"""Prepare an independent-review queue and leakage-safe six-class benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = ROOT / "data" / "evaluation" / "india_gcd_reviewed.csv"
AUTHORED = ROOT / "data" / "evaluation" / "india_gcd_hindi_examples.csv"
REVIEW_QUEUE = ROOT / "data" / "evaluation" / "india_gcd_second_review_queue.csv"
BENCHMARK = ROOT / "data" / "evaluation" / "india_hindi_benchmark_candidates.csv"

DEPARTMENTS = {
    "water_supply": "Water Supply & Sewerage",
    "electricity": "Electricity & Power",
    "roads_transport": "Roads & Transport",
    "healthcare": "Public Health",
    "sanitation": "Environment & Pollution",
    "ration_aadhar": "Public Distribution System",
}

NO_RESPONSE_TEXTS = {
    "Complaints are raised but no action is taken.",
    "We had to wait for hours, and no one responded.",
    "Staff is uncooperative and careless.",
    "The system is always down when we visit the office.",
    "The helpline number is always busy or unreachable.",
    "Despite repeated complaints, the issue is not resolved.",
    "कई बार शिकायत की, लेकिन कोई कार्रवाई नहीं होती।",
    "स्टाफ असहयोगी और लापरवाह है।",
    "लैब जांच की रिपोर्ट बिना कारण देर से मिलती है।",
    "जब भी हम कार्यालय जाते हैं, सिस्टम डाउन रहता है।",
    "हेल्पलाइन नंबर हमेशा व्यस्त या अप्राप्य रहता है।",
    "कई बार शिकायत की, लेकिन कोई समाधान नहीं हुआ।",
}

# These are candidate examples only. Every row remains pending independent review.
PHRASES = {
    "water_supply": {
        "english": [
            "Our lane has had no water supply since Monday.",
            "The tap water is muddy and unsafe to drink.",
            "The water tanker skips our area every week.",
            "Low pressure means the overhead tank never fills.",
        ],
        "hindi": [
            "हमारी गली में सोमवार से पानी नहीं आ रहा है।",
            "नल का पानी गंदा है और पीने लायक नहीं है।",
            "पानी का टैंकर हर हफ्ते हमारे इलाके को छोड़ देता है।",
            "कम दबाव के कारण घर की टंकी नहीं भरती।",
        ],
        "hinglish": [
            "Hamari gali me teen din se paani nahi aa raha.",
            "Nal ka pani ganda hai, drinking ke layak nahi.",
            "Water tanker hamare area me time par nahi aata.",
            "Paani ka pressure bahut low hai, tank nahi bharta.",
        ],
    },
    "electricity": {
        "english": [
            "Power cuts happen every evening in our neighbourhood.",
            "Sparks are coming from the transformer near the school.",
            "The electricity meter bill is incorrect.",
            "The pole wire is hanging dangerously over the road.",
        ],
        "hindi": [
            "हमारे मोहल्ले में हर शाम बिजली कट जाती है।",
            "स्कूल के पास ट्रांसफार्मर से चिंगारी निकलती है।",
            "बिजली के मीटर का बिल गलत बनाया गया है।",
            "खंभे का तार सड़क के ऊपर खतरनाक तरीके से लटक रहा है।",
        ],
        "hinglish": [
            "Hamare area me roz power cut hota hai.",
            "School ke paas transformer se spark nikal raha hai.",
            "Electricity meter ka bill galat aaya hai.",
            "Pole ka wire road ke upar dangerously latak raha hai.",
        ],
    },
    "roads_transport": {
        "english": [
            "Large potholes have made the main road unsafe.",
            "The government bus does not stop at the listed stop.",
            "The auto driver is charging more than the approved fare.",
            "The traffic signal has been broken for two weeks.",
        ],
        "hindi": [
            "बड़े गड्ढों के कारण मुख्य सड़क असुरक्षित हो गई है।",
            "सरकारी बस निर्धारित स्टॉप पर नहीं रुकती।",
            "ऑटो चालक तय किराए से ज्यादा पैसे लेता है।",
            "ट्रैफिक सिग्नल दो हफ्तों से खराब है।",
        ],
        "hinglish": [
            "Main road par itne potholes hain ki drive karna unsafe hai.",
            "Sarkari bus listed stop par rukti hi nahi.",
            "Auto wala approved fare se zyada charge karta hai.",
            "Traffic signal do weeks se kharab pada hai.",
        ],
    },
    "healthcare": {
        "english": [
            "The public hospital has no doctor during evening hours.",
            "The health centre has run out of essential medicines.",
            "An ambulance was not available for an urgent patient.",
            "The vaccination centre has no vaccines for children.",
        ],
        "hindi": [
            "सरकारी अस्पताल में शाम को डॉक्टर उपलब्ध नहीं होते।",
            "स्वास्थ्य केंद्र में जरूरी दवाएं खत्म हो गई हैं।",
            "जरूरी मरीज के लिए एम्बुलेंस उपलब्ध नहीं मिली।",
            "टीकाकरण केंद्र में बच्चों के टीके नहीं हैं।",
        ],
        "hinglish": [
            "Government hospital me evening me doctor nahi milta.",
            "Health centre me essential medicines khatam hain.",
            "Urgent patient ke liye ambulance available nahi thi.",
            "Vaccination centre par bachchon ke vaccines nahi hain.",
        ],
    },
    "sanitation": {
        "english": [
            "Garbage has not been collected from our street for a week.",
            "The open drain smells bad and attracts mosquitoes.",
            "Overflowing bins are spreading waste onto the road.",
            "The cleaning staff has skipped our lane for several days.",
        ],
        "hindi": [
            "एक हफ्ते से हमारी गली का कचरा नहीं उठाया गया।",
            "खुले नाले से बदबू आती है और मच्छर बढ़ रहे हैं।",
            "भरे हुए कूड़ेदान से कचरा सड़क पर फैल रहा है।",
            "सफाई कर्मचारी कई दिनों से हमारी गली में नहीं आया।",
        ],
        "hinglish": [
            "Ek week se hamari street ka garbage collect nahi hua.",
            "Open naali se badbu aa rahi hai aur machhar hain.",
            "Overflowing dustbins ka waste road par fail raha hai.",
            "Cleaning staff kai din se lane me nahi aaya.",
        ],
    },
    "ration_aadhar": {
        "english": [
            "The ration dealer gave less grain than my monthly quota.",
            "A family member's name is wrong on the ration card.",
            "The Aadhaar centre keeps sending us back for corrections.",
            "The biometric machine fails and we cannot collect ration.",
        ],
        "hindi": [
            "राशन डीलर ने महीने के कोटे से कम अनाज दिया।",
            "राशन कार्ड में परिवार के सदस्य का नाम गलत है।",
            "आधार केंद्र पर सुधार के लिए बार-बार वापस भेजते हैं।",
            "बायोमेट्रिक मशीन खराब है और राशन नहीं मिल रहा।",
        ],
        "hinglish": [
            "Ration dealer ne monthly quota se kam anaaj diya.",
            "Ration card me family member ka naam wrong hai.",
            "Aadhaar centre correction ke liye baar baar bula raha hai.",
            "Biometric machine fail hai, ration nahi mil raha.",
        ],
    },
}


def make_candidates() -> pd.DataFrame:
    rows = []
    for category, languages in PHRASES.items():
        for language, seeds in languages.items():
            for seed_index, text in enumerate(seeds):
                for variant in range(3):
                    if language == "english":
                        suffix = (
                            " Please register this complaint."
                            if variant == 0
                            else " Kindly resolve this issue."
                            if variant == 1
                            else " This has continued despite earlier requests."
                        )
                    elif language == "hindi":
                        suffix = (
                            " कृपया इस शिकायत को दर्ज करें।"
                            if variant == 0
                            else " कृपया इसका जल्द समाधान करें।"
                            if variant == 1
                            else " पहले शिकायत करने के बाद भी समस्या जारी है।"
                        )
                    else:
                        suffix = (
                            " Please complaint register kar dijiye."
                            if variant == 0
                            else " Kindly iska solution karwa dijiye."
                            if variant == 1
                            else " Pehle complaint ke baad bhi issue continue hai."
                        )
                    rows.append(
                        {
                            "grievance_text": text + suffix,
                            "source_category": category,
                            "mapped_department": DEPARTMENTS[category],
                            "language": language,
                            "source_file": f"authored_{category}_{language}_{seed_index}_{variant}",
                            "source_group": f"{category}_{language}_{seed_index}",
                            "template_id": f"{category}_{seed_index}",
                            "review_status": "pending_independent_review",
                            "independent_reviewer": "",
                            "independent_review_notes": "",
                            "final_department": "",
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates(subset=["grievance_text", "language"])


def main() -> None:
    reviewed = pd.read_csv(REVIEWED)
    reviewed["independent_reviewer"] = ""
    reviewed["independent_review_notes"] = ""
    reviewed["final_department"] = ""
    reviewed["independent_review_status"] = "pending_independent_review"
    reviewed["source_group"] = reviewed["source_file"].astype(str)
    reviewed["template_id"] = reviewed["source_category"].astype(str)
    queue_columns = [
        "grievance_text", "source_category", "mapped_department", "language",
        "source_file", "source_group", "template_id", "review_status",
        "independent_review_status", "independent_reviewer",
        "independent_review_notes", "final_department", "escalation_flag",
        "escalation_intent", "escalation_reason",
    ]
    reviewed["escalation_flag"] = reviewed["grievance_text"].isin(NO_RESPONSE_TEXTS)
    reviewed["escalation_intent"] = reviewed["escalation_flag"].map(
        {True: "no_response_escalation", False: ""}
    )
    reviewed["escalation_reason"] = reviewed["escalation_flag"].map(
        {
            True: "VOC indicates prior complaint, delayed response, "
            "unresponsive staff/system, or unresolved issue.",
            False: "",
        }
    )
    queue = reviewed[queue_columns]
    REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(REVIEW_QUEUE, index=False, quoting=csv.QUOTE_MINIMAL)

    candidates = make_candidates()
    candidates["independent_review_status"] = "pending_independent_review"
    candidates["escalation_flag"] = False
    candidates["escalation_intent"] = ""
    candidates["escalation_reason"] = ""
    candidates = candidates[queue_columns]
    benchmark = pd.concat([queue, candidates], ignore_index=True)
    benchmark.to_csv(BENCHMARK, index=False, quoting=csv.QUOTE_MINIMAL)
    counts = benchmark.groupby(
        ["mapped_department", "language"], sort=True
    ).size().unstack(fill_value=0)
    print(
        f"Second-review queue: {len(queue)} rows; "
        f"benchmark candidates: {len(benchmark)} rows"
    )
    print(counts.to_string())
    print("All benchmark rows require independent review before evaluation.")


if __name__ == "__main__":
    main()
