"""
=========================================================
SHARED ENUMS / LITERAL TYPES
=========================================================
"""

from typing import Literal

# =========================================================
# RESPONSE TYPE
# =========================================================
ResponseType = Literal["rag", "specific_action", "broadcast_action"]

# =========================================================
# SERVICE TYPE
# =========================================================
ServiceType = Literal[
    "Plumbing", "Electrical", "Carpentry", "Cleaning", "Painting",
    "AC Technician", "Internet Technician", "Appliance Repair",
    "Handyman", "CCTV Installation", "Furniture Moving", "Gardening",
    "Pest Control", "Water Heater Technician", "Satellite Technician",
    "Locksmith", "Gas"
]

# =========================================================
# GOVERNORATE
# =========================================================
Governorate = Literal[
    "Alexandria", "Aswan", "Asyut", "Beheira", "Beni Suef", "Cairo",
    "Dakahlia", "Damietta", "Faiyum", "Gharbia", "Giza", "Ismailia",
    "Kafr El-Sheikh", "Luxor", "Matrouh", "Minya", "Monufia", "New Valley",
    "North Sinai", "Port Said", "Qalyubia", "Qena", "Red Sea",
    "Sharqia", "Sohag", "South Sinai", "Suez"
]

# =========================================================
# CITY
# =========================================================
City = Literal[
    "Abu Qir", "Agami", "Alexandria", "Ar-Raml", "Borg El Arab", "Montaza",
    "New Borg El Arab", "Sidi Bishr", "Abu Simbel", "Aswan", "Idfū",
    "Kawm Umbū", "Abnūb", "Abū Tīj", "Al Badārī", "Al Qūṣīyah", "Asyūṭ",
    "Dayrūṭ", "Manfalūṭ", "Abū al Maṭāmīr", "Ad Dilinjāt", "Damanhūr",
    "Ḥawsh ʿĪsá", "Idkū", "Kafr ad Dawwār", "Kawm Ḥamādah", "Rosetta",
    "Al Fashn", "Banī Suwayf", "Būsh", "Sumusṭā as Sulṭānī", "Badr",
    "Bulaq", "Cairo", "Cairo Downtown", "El Mataria", "Fustat",
    "Hadayek El Kobba", "Heliopolis", "Helwan", "Maadi", "Musturud",
    "Nasr City", "New Administrative Capital of Egypt", "New Cairo", "Rehab",
    "Shubra", "Tura", "Zamalek", "Ajā", "Al Jammālīyah", "Al Manṣūrah",
    "Al Manzalah", "Al Maṭarīyah", "Bilqās", "Dikirnis", "ʿIzbat al Burj",
    "Minyat an Naṣr", "Shirbīn", "Ṭalkhā", "Az Zarqā", "Damietta",
    "Fāraskūr", "Al Fayyūm", "Al Wāsiṭah", "Ibshawāy", "Iṭsā", "Ṭāmiyah",
    "Al Maḥallah al Kubrá", "Basyūn", "Kafr az Zayyāt", "Quṭūr",
    "Samannūd", "Tanda", "Zefta", "Al ʿAyyāṭ", "Al Bawīṭī",
    "Al Ḥawāmidīyah", "Aṣ Ṣaff", "Awsīm", "Giza", "Madīnat Sittah Uktūbar",
    "Ismailia", "Al Ḥāmūl", "Disūq", "Fuwwah", "Kafr ash Shaykh",
    "Markaz Disūq", "Munshāt ʿAlī Āghā", "Sīdī Sālim", "Luxor",
    "Markaz al Uqṣur", "Al ʿAlamayn", "Mersa Matruh", "Siwa Oasis",
    "Abū Qurqāṣ", "Al Minyā", "Banī Mazār", "Dayr Mawās", "Mallawī",
    "Maṭāy", "Samālūṭ", "Al Bājūr", "Ashmūn", "Ash Shuhadāʾ", "Munūf",
    "Quwaysinā", "Shibīn al Kawm", "Talā", "Al Khārijah", "Qaṣr al Farāfirah",
    "Arish", "Port Said", "Al Khānkah", "Al Qanāṭir al Khayrīyah", "Banhā",
    "Obour City", "Qalyūb", "Shibīn al Qanāṭir", "Toukh", "Dishnā",
    "Farshūṭ", "Isnā", "Kousa", "Najaʿ Ḥammādī", "Qinā", "Al Quṣayr",
    "El Gouna", "Hurghada", "Makadi Bay", "Marsa Alam", "Ras Gharib",
    "Safaga", "10th of Ramadan", "Al Qurein", "Awlad Saqr", "Bilbeis",
    "Diyarb Negm", "El Husseiniya", "Faqous", "Hihya", "Kafr Saqr",
    "Markaz Abū Ḥammād", "Mashtoul El Souk", "Minya El Qamh", "New Salhia",
    "Zagazig", "Akhmīm", "Al Balyanā", "Al Manshāh", "Jirjā", "Juhaynah",
    "Markaz Jirjā", "Markaz Sūhāj", "Sohag", "Ṭahṭā", "Dahab", "El-Tor",
    "Nuwaybiʿa", "Saint Catherine", "Sharm el-Sheikh", "Ain Sukhna", "Suez"
]

# =========================================================
# PAYMENT MODE
# =========================================================
PaymentMode = Literal["Fixed Price", "Hourly"]

# =========================================================
# SEARCH SCOPE
# =========================================================
SearchScope = Literal["Governorate", "District"]

# =========================================================
# PLAIN LIST VERSIONS (for runtime use in pipeline.py)
# =========================================================
VALID_SERVICE_TYPES = list(ServiceType.__args__)
VALID_GOVERNORATES  = list(Governorate.__args__)
VALID_CITIES        = list(City.__args__)
VALID_PAYMENT_MODES = list(PaymentMode.__args__)
VALID_SEARCH_SCOPES = list(SearchScope.__args__)
