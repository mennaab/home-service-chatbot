import os
import json
import re
import difflib
import unicodedata
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from app.config import OPENROUTER_API_KEY, MODEL_NAME, EMBEDDING_MODEL

from app.rag.prompts import (
    qa_prompt,
    contextualize_q_prompt,
    ServEaseResponseSchema,
    extraction_prompt,
    BookingSlotsSchema
)

from app.models.enums import (
    VALID_SERVICE_TYPES as ENUM_VALID_SERVICE_TYPES,
    VALID_PAYMENT_MODES as ENUM_VALID_PAYMENT_MODES,
    VALID_SEARCH_SCOPES as ENUM_VALID_SEARCH_SCOPES,
)

# =========================================================
# CITY → GOVERNORATE AUTO-MAPPING
# =========================================================
# If the user gives a city, we fill governorate automatically
# without asking. This removes one question from the flow.

CITY_TO_GOVERNORATE = {
    # Alexandria
    "Abu Qir": "Alexandria", "Agami": "Alexandria", "Alexandria": "Alexandria",
    "Ar-Raml": "Alexandria", "Borg El Arab": "Alexandria", "Montaza": "Alexandria",
    "New Borg El Arab": "Alexandria", "Sidi Bishr": "Alexandria",
    # Aswan
    "Abu Simbel": "Aswan", "Aswan": "Aswan", "Idfū": "Aswan",
    "Kawm Umbū": "Aswan",
    # Asyut
    "Abnūb": "Asyut", "Abū Tīj": "Asyut", "Al Badārī": "Asyut",
    "Al Qūṣīyah": "Asyut", "Asyūṭ": "Asyut", "Dayrūṭ": "Asyut",
    "Manfalūṭ": "Asyut",
    # Beheira
    "Abū al Maṭāmīr": "Beheira", "Ad Dilinjāt": "Beheira",
    "Damanhūr": "Beheira", "Ḥawsh ʿĪsá": "Beheira", "Idkū": "Beheira",
    "Kafr ad Dawwār": "Beheira", "Kawm Ḥamādah": "Beheira", "Rosetta": "Beheira",
    # Beni Suef
    "Al Fashn": "Beni Suef", "Banī Suwayf": "Beni Suef", "Būsh": "Beni Suef",
    "Sumusṭā as Sulṭānī": "Beni Suef",
    # Cairo
    "Badr": "Cairo", "Bulaq": "Cairo", "Cairo": "Cairo",
    "Cairo Downtown": "Cairo", "El Mataria": "Cairo", "Fustat": "Cairo",
    "Hadayek El Kobba": "Cairo", "Heliopolis": "Cairo", "Helwan": "Cairo",
    "Maadi": "Cairo", "Musturud": "Cairo", "Nasr City": "Cairo",
    "New Administrative Capital of Egypt": "Cairo", "New Cairo": "Cairo",
    "Rehab": "Cairo", "Shubra": "Cairo", "Tura": "Cairo", "Zamalek": "Cairo",
    # Dakahlia
    "Ajā": "Dakahlia", "Al Jammālīyah": "Dakahlia", "Al Manṣūrah": "Dakahlia",
    "Al Manzalah": "Dakahlia", "Al Maṭarīyah": "Dakahlia", "Bilqās": "Dakahlia",
    "Dikirnis": "Dakahlia", "ʿIzbat al Burj": "Dakahlia",
    "Minyat an Naṣr": "Dakahlia", "Shirbīn": "Dakahlia", "Ṭalkhā": "Dakahlia",
    # Damietta
    "Az Zarqā": "Damietta", "Damietta": "Damietta", "Fāraskūr": "Damietta",
    # Faiyum
    "Al Fayyūm": "Faiyum", "Al Wāsiṭah": "Faiyum", "Ibshawāy": "Faiyum",
    "Iṭsā": "Faiyum", "Ṭāmiyah": "Faiyum",
    # Gharbia
    "Al Maḥallah al Kubrá": "Gharbia", "Basyūn": "Gharbia",
    "Kafr az Zayyāt": "Gharbia", "Quṭūr": "Gharbia",
    "Samannūd": "Gharbia", "Tanda": "Gharbia", "Zefta": "Gharbia",
    # Giza
    "Al ʿAyyāṭ": "Giza", "Al Bawīṭī": "Giza", "Al Ḥawāmidīyah": "Giza",
    "Aṣ Ṣaff": "Giza", "Awsīm": "Giza", "Giza": "Giza",
    "Madīnat Sittah Uktūbar": "Giza",
    # Ismailia
    "Ismailia": "Ismailia",
    # Kafr El-Sheikh
    "Al Ḥāmūl": "Kafr El-Sheikh", "Disūq": "Kafr El-Sheikh",
    "Fuwwah": "Kafr El-Sheikh", "Kafr ash Shaykh": "Kafr El-Sheikh",
    "Markaz Disūq": "Kafr El-Sheikh", "Munshāt ʿAlī Āghā": "Kafr El-Sheikh",
    "Sīdī Sālim": "Kafr El-Sheikh",
    # Luxor
    "Luxor": "Luxor", "Markaz al Uqṣur": "Luxor",
    # Matrouh
    "Al ʿAlamayn": "Matrouh", "Mersa Matruh": "Matrouh",
    "Siwa Oasis": "Matrouh",
    # Minya
    "Abū Qurqāṣ": "Minya", "Al Minyā": "Minya", "Banī Mazār": "Minya",
    "Dayr Mawās": "Minya", "Mallawī": "Minya", "Maṭāy": "Minya",
    "Samālūṭ": "Minya",
    # Monufia
    "Al Bājūr": "Monufia", "Ashmūn": "Monufia", "Ash Shuhadāʾ": "Monufia",
    "Munūf": "Monufia", "Quwaysinā": "Monufia",
    "Shibīn al Kawm": "Monufia", "Talā": "Monufia",
    # New Valley
    "Al Khārijah": "New Valley", "Qaṣr al Farāfirah": "New Valley",
    # North Sinai
    "Arish": "North Sinai",
    # Port Said
    "Port Said": "Port Said",
    # Qalyubia
    "Al Khānkah": "Qalyubia", "Al Qanāṭir al Khayrīyah": "Qalyubia",
    "Banhā": "Qalyubia", "Obour City": "Qalyubia", "Qalyūb": "Qalyubia",
    "Shibīn al Qanāṭir": "Qalyubia", "Toukh": "Qalyubia",
    # Qena
    "Dishnā": "Qena", "Farshūṭ": "Qena", "Isnā": "Qena",
    "Kousa": "Qena", "Najaʿ Ḥammādī": "Qena", "Qinā": "Qena",
    # Red Sea
    "Al Quṣayr": "Red Sea", "El Gouna": "Red Sea", "Hurghada": "Red Sea",
    "Makadi Bay": "Red Sea", "Marsa Alam": "Red Sea", "Ras Gharib": "Red Sea",
    "Safaga": "Red Sea",
    # Sharqia
    "10th of Ramadan": "Sharqia", "Al Qurein": "Sharqia",
    "Awlad Saqr": "Sharqia", "Bilbeis": "Sharqia", "Diyarb Negm": "Sharqia",
    "El Husseiniya": "Sharqia", "Faqous": "Sharqia", "Hihya": "Sharqia",
    "Kafr Saqr": "Sharqia", "Markaz Abū Ḥammād": "Sharqia",
    "Mashtoul El Souk": "Sharqia", "Minya El Qamh": "Sharqia",
    "New Salhia": "Sharqia", "Zagazig": "Sharqia",
    # Sohag
    "Akhmīm": "Sohag", "Al Balyanā": "Sohag", "Al Manshāh": "Sohag",
    "Jirjā": "Sohag", "Juhaynah": "Sohag", "Markaz Jirjā": "Sohag",
    "Markaz Sūhāj": "Sohag", "Sohag": "Sohag", "Ṭahṭā": "Sohag",
    # South Sinai
    "Dahab": "South Sinai", "El-Tor": "South Sinai",
    "Nuwaybiʿa": "South Sinai", "Saint Catherine": "South Sinai",
    "Sharm el-Sheikh": "South Sinai",
    # Suez
    "Ain Sukhna": "Suez", "Suez": "Suez",
}


def infer_governorate_from_city(city: str) -> str | None:
    """Return the governorate for a given city, or None if unknown."""
    if not city:
        return None
    return CITY_TO_GOVERNORATE.get(city)


# =========================================================
# CANONICAL VALUE NORMALIZATION
# =========================================================
# المشكلة: extraction_text_chain (الـ slot extraction) بترجع نص حر من غير
# قيود Literal، فممكن الموديل يكتب "Shebin El Kom" بدل "Shibīn al Kawm".
# الحل: أي قيمة جاية من الـ extraction، نمررها على match_to_canonical()
# قبل ما تدخل الـ data dict، فإما تتحول للشكل الصحيح المطابق للـ Literal،
# أو ترجع None (أحسن من قيمة غلط تكسر الباكند بتاع الحجز).
#
# VALID_CITIES / VALID_GOVERNORATES تتبني من CITY_TO_GOVERNORATE فوق (مصدرها
# مرتبط بمنطق auto-fill الخاص بيها). باقي اللستات بتتسحب من enums.py المركزي
# عشان تفضل متطابقة دايمًا مع ServEaseResponseSchema/BookingSlotsSchema/ChatData
# من غير ما تتكرر يدويًا في أكتر من مكان.

VALID_CITIES = list(CITY_TO_GOVERNORATE.keys())
VALID_GOVERNORATES = sorted(set(CITY_TO_GOVERNORATE.values()))
VALID_SERVICE_TYPES = ENUM_VALID_SERVICE_TYPES
VALID_PAYMENT_MODES = ENUM_VALID_PAYMENT_MODES
VALID_SEARCH_SCOPES = ENUM_VALID_SEARCH_SCOPES

# Aliases يدوية للأسماء اللي شكلها مختلف تمامًا عن الكانونيكال
# (مش مجرد فرق في الـ diacritics، ده فرق في الكتابة نفسها)
CITY_ALIASES = {
    "shebin el kom": "Shibīn al Kawm",
    "shebeen el kom": "Shibīn al Kawm",
    "shebin elkom": "Shibīn al Kawm",
    "mansoura": "Al Manṣūrah",
    "el mansoura": "Al Manṣūrah",
    "el-mansoura": "Al Manṣūrah",
    "alex": "Alexandria",
    "alexandria city": "Alexandria",
    "sharm": "Sharm el-Sheikh",
    "sharm el sheikh": "Sharm el-Sheikh",
    "sharm al sheikh": "Sharm el-Sheikh",
    "6th of october": "Madīnat Sittah Uktūbar",
    "6 october": "Madīnat Sittah Uktūbar",
    "october city": "Madīnat Sittah Uktūbar",
    "new capital": "New Administrative Capital of Egypt",
    "the new capital": "New Administrative Capital of Egypt",
    "new admin capital": "New Administrative Capital of Egypt",
    "marsa matrouh": "Mersa Matruh",
    "matrouh": "Mersa Matruh",
    "marsa alam city": "Marsa Alam",
    "tanta": None,  # مش موجودة في القايمة الكانونيكال أصلاً
}

GOVERNORATE_ALIASES = {
    "kafr elsheikh": "Kafr El-Sheikh",
    "kafr el sheikh": "Kafr El-Sheikh",
    "beni-suef": "Beni Suef",
    "bani suwayf": "Beni Suef",
    "el sharqia": "Sharqia",
    "sharkia": "Sharqia",
    "qaliubiya": "Qalyubia",
    "qalyoubia": "Qalyubia",
    "monofia": "Monufia",
    "menofia": "Monufia",
}

SERVICE_TYPE_ALIASES = {
    "plumber": "Plumbing",
    "electrician": "Electrical",
    "carpenter": "Carpentry",
    "ac technician": "AC Technician",
    "ac repair": "AC Technician",
    "air conditioning": "AC Technician",
    "internet": "Internet Technician",
    "wifi technician": "Internet Technician",
    "appliance repair": "Appliance Repair",
    "appliances": "Appliance Repair",
    "cctv": "CCTV Installation",
    "camera installation": "CCTV Installation",
    "moving": "Furniture Moving",
    "movers": "Furniture Moving",
    "gardener": "Gardening",
    "pest control": "Pest Control",
}

PAYMENT_MODE_ALIASES = {
    "fixed": "Fixed Price",
    "fixed price": "Fixed Price",
    "per hour": "Hourly",
    "hourly rate": "Hourly",
}

SEARCH_SCOPE_ALIASES = {
    "governorate": "Governorate",
    "whole governorate": "Governorate",
    "all governorate": "Governorate",
    "المحافظة": "Governorate",
    "كل المحافظة": "Governorate",
    "district": "District",
    "area": "District",
    "my area": "District",
    "منطقتي": "District",
    "نفس المنطقة": "District",
}


def _strip_diacritics(text: str) -> str:
    """Remove Latin transliteration diacritics (ā, ī, ṭ, ʿ, etc.) for comparison."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_for_matching(text: str) -> str:
    """Lowercase, strip diacritics/punctuation, drop Arabic articles (al/el/ad/as/az/ash)."""
    text = _strip_diacritics(text).lower()
    text = text.replace("-", " ").replace("'", "").replace("ʿ", "").replace("ʾ", "")
    text = re.sub(r"\b(al|el|ad|as|az|ash)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_to_canonical(
    value: str | None,
    valid_options: list[str],
    manual_aliases: dict | None = None,
    field_name: str = "",
    fuzzy_cutoff: float = 0.75
) -> str | None:
    """
    Map a free-text value extracted by the LLM to the EXACT canonical string
    expected by the Literal schema / downstream booking backend.
    Returns None (never a guessed/wrong value) if no confident match is found.
    """
    if not value:
        return None

    raw = value.strip()

    # 1) Already exact
    if raw in valid_options:
        return raw

    normalized = _normalize_for_matching(raw)

    # 2) Manual alias table (handles spellings that diverge a lot, not just diacritics)
    if manual_aliases and normalized in manual_aliases:
        return manual_aliases[normalized]  # may be None on purpose (e.g. "tanta")

    # 3) Diacritic/case-insensitive exact match
    lookup = {_normalize_for_matching(opt): opt for opt in valid_options}
    if normalized in lookup:
        return lookup[normalized]

    # 4) Fuzzy fallback for small typos / minor spelling drift
    close = difflib.get_close_matches(normalized, lookup.keys(), n=1, cutoff=fuzzy_cutoff)
    if close:
        matched = lookup[close[0]]
        print(f"[NORMALIZE] field={field_name} '{value}' -> '{matched}' (fuzzy match)")
        return matched

    # 5) No confident match — drop it rather than send a value that crashes the backend
    print(f"[NORMALIZE] field={field_name} could not match '{value}' to any known value, dropping it")
    return None


def normalize_city(value):
    return match_to_canonical(value, VALID_CITIES, CITY_ALIASES, field_name="city")


def normalize_governorate(value):
    return match_to_canonical(value, VALID_GOVERNORATES, GOVERNORATE_ALIASES, field_name="governorate")


def normalize_service_type(value):
    return match_to_canonical(value, VALID_SERVICE_TYPES, SERVICE_TYPE_ALIASES, field_name="service_type")


def normalize_payment_mode(value):
    return match_to_canonical(value, VALID_PAYMENT_MODES, PAYMENT_MODE_ALIASES, field_name="payment_mode")


def normalize_search_scope(value):
    return match_to_canonical(value, VALID_SEARCH_SCOPES, SEARCH_SCOPE_ALIASES, field_name="search_scope")


# =========================================================
# RELATIVE DATE RESOLUTION
# =========================================================
# المشكلة: الـ LLM بيستخرج preferred_date كنص حر زي "tomorrow"/"بكرة"،
# لكن الـ UI (Date Picker) محتاج تاريخ فعلي YYYY-MM-DD عشان يحدده في
# الكالندر، فلو بعتنا "tomorrow" كنص، الباكند/الـ UI بيعمل error.
# الحل: نحول أي تعبير نسبي لتاريخ فعلي بنحسبه من تاريخ اليوم وقت الريكويست.

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# اسم اليوم -> رقمه (الاتنين = 0 ... الحد = 6) — نفس ترتيب Python weekday()
WEEKDAY_NAMES = {
    "monday": 0, "الاتنين": 0, "الإتنين": 0, "اتنين": 0,
    "tuesday": 1, "التلات": 1, "الثلاثاء": 1, "تلات": 1,
    "wednesday": 2, "الاربع": 2, "الأربعاء": 2, "اربع": 2,
    "thursday": 3, "الخميس": 3, "خميس": 3,
    "friday": 4, "الجمعة": 4, "جمعة": 4, "جمعه": 4,
    "saturday": 5, "السبت": 5, "سبت": 5,
    "sunday": 6, "الحد": 6, "الأحد": 6, "حد": 6,
}


def resolve_relative_date(date_str: str | None, reference: datetime | None = None) -> str | None:
    """
    Convert a free-text relative date expression (as extracted by the LLM,
    e.g. 'tomorrow', 'بكرة', 'in a week', 'كمان 3 ايام', 'الجمعة') into a
    concrete ISO date string (YYYY-MM-DD) the booking UI's date picker expects.

    If the value is already a concrete date, or can't be confidently
    resolved, it's returned UNCHANGED (never guessed) — the caller can
    decide what to do with an unresolved value.
    """
    if not date_str:
        return date_str

    today = (reference or datetime.now()).date()
    raw = date_str.strip()
    text = raw.lower().translate(_ARABIC_DIGITS)

    # 1) Already a concrete ISO date -> leave as-is
    if re.match(r"^\d{4}-\d{1,2}-\d{1,2}$", text):
        return text

    # 2) Already a concrete D/M/Y or D-M-Y date -> leave as-is
    if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", text):
        return text

    # 3) Today  (including compound like "النهاردة بليل" / "today evening")
    if re.search(r"\b(today|اليوم|النهاردة|النهارده|دلوقتي)\b", text):
        return today.isoformat()

    # 4) Day after tomorrow — MUST be checked BEFORE tomorrow
    if re.search(r"(day after tomorrow|بعد بكرة|بعد بكره|بعد غد\b)", text):
        return (today + timedelta(days=2)).isoformat()

    # 5) Tomorrow  (including compound "بكرة الصبح" / "tomorrow morning")
    if re.search(r"\b(tomorrow|بكرة|بكره|غدا|غداً|غدًا)\b", text):
        return (today + timedelta(days=1)).isoformat()

    # 6) "next week" / "الأسبوع الجاي" — means the SAME weekday next week
    #    (not +7 blindly, but the matching weekday 7 days from today's weekday)
    #    We detect this BEFORE the general "N weeks" pattern.
    if re.search(r"\b(next week|الاسبوع الجاي|الأسبوع الجاي|الاسبوع الجاية|أول الأسبوع الجاي|اول الاسبوع الجاي)\b", text):
        # "next week" with no specific weekday → Monday of next week
        # "أول الأسبوع الجاي" → Monday of next week
        if re.search(r"(أول|اول|first|start)", text):
            # next Monday
            days_ahead = (0 - today.weekday() + 7) % 7
            days_ahead = days_ahead or 7
            return (today + timedelta(days=days_ahead)).isoformat()
        # plain "next week" → +7 days from today
        return (today + timedelta(weeks=1)).isoformat()

    # 7) "end of week" / "آخر الأسبوع" → next Friday
    if re.search(r"(end of (the )?week|آخر الاسبوع|آخر الأسبوع|نهاية الاسبوع|نهاية الأسبوع|weekend)", text):
        days_ahead = (4 - today.weekday() + 7) % 7   # 4 = Friday in Python weekday()
        days_ahead = days_ahead or 7
        return (today + timedelta(days=days_ahead)).isoformat()

    # 8) "next month" / "الشهر الجاي" / "بعد شهر" / "in a month"
    if re.search(r"(next month|الشهر الجاي|الشهر الجاية|بعد شهر|in a month|in one month)", text):
        # Same day next month (clamp if month is shorter)
        month = today.month + 1
        year = today.year
        if month > 12:
            month = 1
            year += 1
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        day = min(today.day, last_day)
        from datetime import date as date_cls
        return date_cls(year, month, day).isoformat()

    # 9) "in N months" / "بعد N شهر"
    month_match = (
        re.search(r"(\d+)\s*month", text)
        or re.search(r"بعد\s*(\d+)\s*(شهر|أشهر|اشهر)", text)
        or re.search(r"كمان\s*(\d+)\s*(شهر|أشهر|اشهر)", text)
    )
    if month_match:
        n = int(month_match.group(1))
        month = today.month + n
        year = today.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        day = min(today.day, last_day)
        from datetime import date as date_cls
        return date_cls(year, month, day).isoformat()

    # 10) "in N weeks" / "كمان N اسبوع" / "بعد اسبوع"
    week_match = (
        re.search(r"(\d+)\s*week", text)
        or re.search(r"كمان\s*(\d+)\s*(اسبوع|أسبوع)", text)
        or re.search(r"بعد\s*(\d+)\s*(اسبوع|أسبوع)", text)
    )
    if week_match:
        weeks = int(week_match.group(1))
        return (today + timedelta(weeks=weeks)).isoformat()

    # 11) "in N days" / "كمان N يوم/أيام" / "بعد N يوم"
    day_match = (
        re.search(r"(\d+)\s*day", text)
        or re.search(r"كمان\s*(\d+)\s*(يوم|ايام|أيام)", text)
        or re.search(r"بعد\s*(\d+)\s*(يوم|ايام|أيام)", text)
    )
    if day_match:
        days = int(day_match.group(1))
        return (today + timedelta(days=days)).isoformat()

    # 12) Verbal date: "25 يونيو" / "June 25" / "25 June" / "25/6" with no year
    MONTH_NAMES = {
        "january": 1, "jan": 1, "يناير": 1,
        "february": 2, "feb": 2, "فبراير": 2,
        "march": 3, "mar": 3, "مارس": 3,
        "april": 4, "apr": 4, "أبريل": 4, "ابريل": 4,
        "may": 5, "مايو": 5,
        "june": 6, "jun": 6, "يونيو": 6, "يونية": 6,
        "july": 7, "jul": 7, "يوليو": 7, "يولية": 7,
        "august": 8, "aug": 8, "أغسطس": 8, "اغسطس": 8,
        "september": 9, "sep": 9, "sept": 9, "سبتمبر": 9,
        "october": 10, "oct": 10, "أكتوبر": 10, "اكتوبر": 10,
        "november": 11, "nov": 11, "نوفمبر": 11,
        "december": 12, "dec": 12, "ديسمبر": 12,
    }
    for month_name, month_num in MONTH_NAMES.items():
        # Pattern: "25 june" or "june 25"
        m = re.search(rf"(\d{{1,2}})\s*{re.escape(month_name)}", text) or \
            re.search(rf"{re.escape(month_name)}\s*(\d{{1,2}})", text)
        if m:
            day_num = int(m.group(1))
            year = today.year
            from datetime import date as date_cls
            target = date_cls(year, month_num, day_num)
            # If the date has already passed this year, assume next year
            if target < today:
                target = date_cls(year + 1, month_num, day_num)
            return target.isoformat()

    # 13) Weekday name (plain or with "next week" qualifier already consumed above)
    #     "الأسبوع الجاي السبت" → Saturday next week (not nearest Saturday)
    is_next_week_qualified = re.search(
        r"(next week|الاسبوع الجاي|الأسبوع الجاي|الاسبوع الجاية)", text
    )
    for name, weekday_idx in WEEKDAY_NAMES.items():
        if re.search(rf"\b{re.escape(name)}\b", text) or name in text:
            days_ahead = (weekday_idx - today.weekday() + 7) % 7
            if is_next_week_qualified:
                # Force it to the NEXT week's occurrence, not the nearest
                days_ahead = days_ahead or 7
                if days_ahead < 7:
                    days_ahead += 7
            else:
                # Nearest future occurrence (skip today itself)
                days_ahead = days_ahead or 7
            return (today + timedelta(days=days_ahead)).isoformat()

    # 14) Couldn't confidently resolve — return the original value unchanged,
    # never guess. Caller decides what to do (e.g. log it, or re-ask user).
    print(f"[DATE_RESOLVE] could not resolve '{date_str}' to a concrete date, returning as-is")
    return raw


# =========================================================
# VECTOR DB
# =========================================================

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = Chroma(
    persist_directory="db/chroma",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 5})

# =========================================================
# LLM
# =========================================================
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=500
)

# =========================================================
# REPHRASE CHAIN
# =========================================================

rephrase_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
)

# =========================================================
# STRUCTURED OUTPUT
# =========================================================

structured_llm = llm.with_structured_output(
    ServEaseResponseSchema,
    method="function_calling"
)

# =========================================================
# SLOT EXTRACTION (plain-text JSON)
# =========================================================

extraction_text_chain = extraction_prompt | llm | StrOutputParser()


def extract_json_from_text(text: str) -> dict:
    """Pull the first {...} JSON object out of a text blob and parse it."""
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# =========================================================
# HELPERS
# =========================================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


question_answer_chain = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "language": lambda x: x["language"],
        "context": lambda x: format_docs(x["context"]),
    }
    | qa_prompt
    | structured_llm
)

# =========================================================
# LANGUAGE DETECTOR
# =========================================================

def detect_language(text: str) -> str:
    for char in text:
        if "\u0600" <= char <= "\u06FF":
            return "Arabic"
    return "English"


# =========================================================
# HARD GUARD
# =========================================================

REQUIRED_BOOKING_FIELDS = [
    "service_type",
    "governorate",
    "city",
    "street",
    "exact_location",
    "preferred_date",
    "preferred_time",
    "payment_mode",
    # preferred_price and search_scope are intentionally excluded — both are
    # optional in the UI (user may not have a fixed budget or scope preference)
]


def enforce_action_guard(data: dict, language: str) -> dict:
    if data.get("response_type") not in ("specific_action", "broadcast_action"):
        return data

    missing = [f for f in REQUIRED_BOOKING_FIELDS if not data.get(f)]

    if data.get("response_type") == "specific_action" and not data.get("provider_name"):
        missing.append("provider_name")

    if missing:
        print(f"[GUARD] Blocked premature '{data.get('response_type')}' — missing: {missing}")
        data["response_type"] = "rag"
        # IMPORTANT: always overwrite text_response here, not just when empty.
        # The LLM may have already written a "Booking confirmed..." style message
        # believing all fields were filled, while this guard caught a field it
        # missed (e.g. exact_location). If we only filled text_response when
        # empty, that false confirmation message would leak straight to the user
        # even though response_type was correctly downgraded to 'rag' — telling
        # them their booking is done when it isn't.
        first_missing = missing[0]
        ask_again = {
            "provider_name": "ممكن اسم الفني اللي تحب تحجز معاه؟" if language == "Arabic" else "What's the name of the provider you'd like to book?",
            "service_type": "تحب تحجز أنهي خدمة بالظبط؟" if language == "Arabic" else "Which service would you like to book?",
            "governorate": "في أنهي محافظة؟" if language == "Arabic" else "Which governorate is this for?",
            "city": "في أنهي مدينة أو حي؟" if language == "Arabic" else "Which city or area?",
            "street": "ممكن اسم الشارع؟" if language == "Arabic" else "What's the street?",
            "exact_location": "ممكن توضحلي تفاصيل أكتر عن المكان بالظبط زي رقم المبنى، الدور، ورقم الشقة؟" if language == "Arabic" else "Could you give me more precise location details, like the building number, floor, and apartment?",
            "preferred_date": "تحب الميعاد يكون إمتى؟" if language == "Arabic" else "What date works for you?",
            "preferred_time": "في أنهي وقت؟" if language == "Arabic" else "What time works for you?",
            "payment_mode": "هتفضل سعر ثابت للخدمة كلها ولا سعر بالساعة؟" if language == "Arabic" else "Would you prefer a fixed price for the whole job or an hourly rate?",
        }
        data["text_response"] = ask_again.get(
            first_missing,
            (
                "محتاج كام تفصيلة كمان قبل ما أقدر أكمل الحجز."
                if language == "Arabic"
                else "I still need a few more details before I can proceed with the booking."
            )
        )

    return data


def merge_extracted_slots(data: dict, extracted_slots: dict) -> dict:
    """
    Overlay extracted slot values onto the response dict.
    Also auto-fills governorate from city if governorate is still missing.
    """
    if not extracted_slots:
        # Even with no new extraction, try to infer governorate from existing city
        if not data.get("governorate") and data.get("city"):
            inferred = infer_governorate_from_city(data["city"])
            if inferred:
                data["governorate"] = inferred
                print(f"[AUTO] governorate inferred from city: {data['city']} → {inferred}")
        return data

    field_map = {
        "service_type": "service_type",
        "provider_name": "provider_name",
        "governorate": "governorate",
        "city": "city",
        "street": "street",
        "exact_location": "exact_location",
        "preferred_date": "preferred_date",
        "preferred_time": "preferred_time",
        "payment_mode": "payment_mode",
        "preferred_price": "preferred_price",
        "search_scope": "search_scope",
    }

    # حقول لازم تتطابق Literal بالظبط — بتعدي على normalize قبل ما تدخل الـ data
    NORMALIZERS = {
        "city": normalize_city,
        "governorate": normalize_governorate,
        "service_type": normalize_service_type,
        "payment_mode": normalize_payment_mode,
        "preferred_date": resolve_relative_date,
        "search_scope": normalize_search_scope,
    }

    for slot_key, data_key in field_map.items():
        value = extracted_slots.get(slot_key)
        if value is None:
            continue

        normalizer = NORMALIZERS.get(data_key)
        if normalizer:
            normalized_value = normalizer(value)
            if normalized_value is None:
                # ماقدرناش نتأكد من القيمة — نسيب الحقل زي ما هو (أو فاضي)
                # بدل ما نبعت قيمة هتكسر الباكند بتاع الحجز
                continue
            value = normalized_value

        data[data_key] = value

    # AUTO-FILL governorate from city if still missing after extraction
    if not data.get("governorate") and data.get("city"):
        inferred = infer_governorate_from_city(data["city"])
        if inferred:
            data["governorate"] = inferred
            print(f"[AUTO] governorate inferred from city: {data['city']} → {inferred}")

    return data


# =========================================================
# MAIN ASK FUNCTION
# =========================================================

def ask(user_message: str, chat_history: list) -> dict:

    # STEP 1: Detect language
    language = detect_language(user_message)

    # STEP 2: Rephrase / classify
    new_question = rephrase_chain.invoke({
        "input": user_message,
        "chat_history": chat_history
    })

    # STEP 3: Check prefix
    is_action = new_question.strip().startswith("[ACTION]")
    is_continue = new_question.strip().startswith("[CONTINUE]")

    if is_action:
        new_question = new_question.replace("[ACTION]", "").strip()
        retrieved_docs = []
    elif is_continue:
        raw_answer = new_question.replace("[CONTINUE]", "").strip()
        new_question = (
            "[The user is continuing an active booking request and is answering "
            f"the assistant's previous question. Their answer is: \"{raw_answer}\". "
            "Use this answer to fill in the relevant booking field and proceed "
            "with the booking flow — do NOT treat it as a general knowledge question.]"
        )
        retrieved_docs = []
    else:
        retrieved_docs = retriever.invoke(new_question)

    # STEP 4: Slot extraction (only during booking flow)
    extracted_slots = {}
    extraction_debug_info = ""
    if is_action or is_continue:
        try:
            raw_text = extraction_text_chain.invoke({
                "input": new_question,
                "chat_history": chat_history
            })
            extraction_debug_info = f"raw text: {raw_text}"
            print(f"[DEBUG] extraction raw text: {raw_text}")
            parsed = extract_json_from_text(raw_text)
            extracted_slots = {k: v for k, v in parsed.items() if v is not None}
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            extraction_debug_info = f"EXCEPTION: {e}\n{tb}"
            print(f"[WARN] slot extraction failed: {e}")
            extracted_slots = {}

    # STEP 5: Debug file
    with open("retrieved_chunks.txt", "w", encoding="utf-8") as f:
        f.write("DETECTED LANGUAGE: " + language + "\n\n")
        f.write("ORIGINAL QUESTION:\n")
        f.write(user_message)
        f.write("\n\n")
        f.write("IS ACTION: " + str(is_action) + "\n")
        f.write("IS CONTINUE: " + str(is_continue) + "\n\n")
        f.write("REWRITTEN QUESTION:\n")
        f.write(str(new_question))
        f.write("\n\n")
        f.write("EXTRACTED SLOTS:\n")
        f.write(str(extracted_slots))
        f.write("\n\n")
        f.write("EXTRACTION DEBUG INFO:\n")
        f.write(extraction_debug_info)
        f.write("\n\n")
        f.write("=" * 50 + "\n")
        f.write("RETRIEVED CHUNKS\n")
        f.write("=" * 50 + "\n\n")
        if not retrieved_docs:
            f.write("NO CHUNKS RETRIEVED\n")
        else:
            for i, doc in enumerate(retrieved_docs):
                f.write("CHUNK " + str(i+1) + ":\n\n")
                f.write(doc.page_content)
                f.write("\n\n")
                f.write("-" * 50 + "\n\n")

    # STEP 6: Generate structured response
    try:
        response = question_answer_chain.invoke({
            "input": new_question,
            "chat_history": chat_history,
            "context": retrieved_docs,
            "language": language
        })
    except Exception as e:
        print(f"[ERROR] structured_llm invocation failed: {e}")
        fallback = ServEaseResponseSchema(
            response_type="rag",
            text_response=(
                "معلش، حصلت مشكلة فنية وأنا بحاول أجاوبك. ممكن تجرب تبعت سؤالك تاني؟"
                if language == "Arabic"
                else "Sorry, something went wrong. Could you try again?"
            )
        )
        return fallback.model_dump()

    # STEP 7: Normalize, merge slots, auto-fill, guard
    print("=" * 50)
    print(type(response))
    print(response)
    print("=" * 50)

    if hasattr(response, "model_dump"):
        data = response.model_dump()
        data = merge_extracted_slots(data, extracted_slots)
        data = enforce_action_guard(data, language)
        print("FINAL DATA:", data)
        return data

    elif isinstance(response, dict):
        # دفاع إضافي: لو الـ dict الخام فيه قيمة مش متطابقة 100% مع الـ Literal
        # (زي city بشكل مختلف شوية)، نطبّعها الأول قبل المحاولة، بدل ما نرمي
        # كل الـ response في fallback لمجرد فرق إملائي بسيط في حقل واحد.
        for key, normalizer in (
            ("city", normalize_city),
            ("governorate", normalize_governorate),
            ("service_type", normalize_service_type),
            ("payment_mode", normalize_payment_mode),
            ("search_scope", normalize_search_scope),
        ):
            if response.get(key):
                response[key] = normalizer(response[key])

        try:
            validated = ServEaseResponseSchema(**response)
            data = validated.model_dump()
            data = merge_extracted_slots(data, extracted_slots)
            data = enforce_action_guard(data, language)
            return data
        except Exception as e:
            print(f"[ERROR] dict validation failed: {e}")
            fallback = ServEaseResponseSchema(
                response_type="rag",
                text_response=response.get("text_response") or (
                    "معلش، حصلت مشكلة فنية."
                    if language == "Arabic"
                    else "Sorry, something went wrong."
                )
            )
            return fallback.model_dump()

    else:
        return ServEaseResponseSchema(
            text_response=str(response),
            response_type="rag"
        ).model_dump()
