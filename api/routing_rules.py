"""High-signal routing vocabulary shared by the API and regression tests."""

INDIA_RULES = {
    "Water Supply & Sewerage": (
        "water supply", "drinking water", "handpump", "sewage",
        "sewer", "drainage", "water tanker", "water connection",
        "nal ka paani", "nal ka pani", "peene ka pani",
        "peene ka paani", "paani nhi", "pani nhi", "paani nahi",
        "pani nahi", "naali overflow", "ganda paani", "ganda pani",
    ),
    "Roads & Transport": (
        "pothole", "traffic signal", "bus stop", "road", "highway",
        "traffic", "public transport", "street crossing", "parked",
        "parking", "double parked", "double-parked", "blocked road",
        "blocked roadway", "no access", "cannot get out",
        "nikalne ki jagah", "raasta", "gadiya", "gaadi",
    ),
    "Electricity & Power": (
        "power cut", "power outage", "electricity", "transformer",
        "meter", "fallen power line", "electric wire",
    ),
    "Public Health": (
        "hospital", "health centre", "health center", "ambulance",
        "doctor", "medicine", "medical", "clinic", "accident",
        "bleeding", "blood", "blood loss", "khoon", "zakhmi",
        "injured", "insaan ka khoon",
    ),
    "Environment & Pollution": (
        "pollution", "plastic waste", "garbage", "industrial discharge",
        "waste burning", "mosquito", "contamination", "factory ka kala dhuan",
        "factory ka kala dhuaan", "dhuan", "dhuaan", "badbu",
        "saans lene me dikkat", "kachra", "plastic jama", "machhar",
    ),
    "Police & Public Safety": (
        "police", "stolen", "crime", "violent", "chain snatching",
        "unsafe", "attack", "law and order",
    ),
    "Women & Child Welfare": (
        "domestic violence", "child labour", "child labor", "anganwadi",
        "women protection", "child protection", "shelter",
        "maar-peet", "maar pit", "mahila ko ghar", "bachcha dara",
    ),
    "Social Welfare": (
        "pension", "elderly", "disability certificate", "welfare",
        "social security", "old age", "food or medicines",
        "meri pension", "pension nahi", "pension nhi",
    ),
    "Education": (
        "school", "student", "scholarship", "teacher", "classroom",
        "toilet in school", "education", "scholarship ka paisa",
        "student ke account", "status pending",
    ),
    "Municipal Services": (
        "birth certificate", "property tax", "street cleaning",
        "municipal office", "drain maintenance", "civic",
        "janam praman patra", "naali safai", "sadak par jama",
        "property tax", "arrears", "receipt number", "street light",
    ),
    "Revenue & Land Records": (
        "land mutation", "land record", "revenue record", "tehsil",
        "encroachment", "property record", "survey",
    ),
    "Agriculture & Rural Development": (
        "farmer", "crops", "irrigation canal", "seed", "agriculture",
        "harvest", "village irrigation", "khet", "fasal", "keede",
        "sinchai", "nahar me paani", "gaon ki fasal",
    ),
    "Public Distribution System": (
        "ration shop", "ration card", "food grains", "fair price shop",
        "subsidised", "subsidized", "kerosene quota", "ration card",
        "naam galat", "correction",
    ),
    "Non-Complaint": (
        "general information", "please explain", "how to apply",
        "required documents", "office address", "working hours",
    ),
}

MEDICAL_EMERGENCY_TERMS = (
    "accident", "blood loss", "severe bleeding", "heavy bleeding",
    "bleeding heavily", "khoon", "bht zyada chot", "bht zada chot",
    "bahut zyada chot", "zakhmi", "injured", "unconscious",
    "aadmi dab gaya", "insaan dab gaya", "ambulance",
)
