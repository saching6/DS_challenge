import re
import unicodedata
import pandas as pd


## AI generated french dictionaries (Manually verified before submission)
FR_MONTHS = {
    'janv': 1, 'jan': 1, 'fevr': 2, 'févr': 2, 'fev': 2, 'mars': 3, 'avri': 4,
    'avr': 4, 'mai': 5, 'juin': 6, 'juil': 7, 'aout': 8, 'août': 8, 'aou': 8,
    'sept': 9, 'sep': 9, 'octo': 10, 'oct': 10, 'nove': 11, 'nov': 11,
    'dece': 12, 'déce': 12, 'dec': 12, 'déc': 12,
}

EN_MONTHS = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

NA_TOKENS = {"unassigned", "N/A"}
def apply_map(series, mapper):
    lookup = None if callable(mapper) else {str(k): v for k, v in mapper.items()}
    def one(x):
        if x is None or pd.isna(x) or x in NA_TOKENS:
            return x
        if lookup is None:
            return mapper(x)
        if isinstance(x, float) and x.is_integer():   # 1.0 -> "1", not "1.0"
            x = int(x)
        return lookup.get(str(x), pd.NA)
    return series.map(one)


def remove_accents(input_str: str) -> str:
    """Removes accents: 'févr' -> 'fevr', 'août' -> 'aout', 'déc' -> 'dec'."""
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def map_line_colors(text: str) -> str:
    """'Ligne 1, 4, 5' -> 'green_yellow_blue'; 'Ligne verte' -> 'green'."""
    line_colors = {1: 'green', 2: 'orange', 4: 'yellow', 5: 'blue'}
    french_colors = {
        'verte': 'green', 'vert': 'green', 'orange': 'orange',
        'jaune': 'yellow', 'bleue': 'blue', 'bleu': 'blue',
    }
    color_order = ['green', 'orange', 'yellow', 'blue']

    text_lower = str(text).lower()

    for fr_color, en_color in french_colors.items():
        if fr_color in text_lower:
            return en_color

    # Excel turns "2-4" into "02-avr"; recover the month as the second line.
    excel = re.match(r'^\s*(\d{1,2})[-/]([a-zéûà]+)\.?\s*$', text_lower)
    if excel:
        month = FR_MONTHS.get(excel.group(2)[:4]) or FR_MONTHS.get(excel.group(2)[:3])
        numbers = [excel.group(1)] + ([str(month)] if month else [])
    else:
        numbers = re.findall(r'\d+', text_lower)

    found = {line_colors[int(n)] for n in numbers if int(n) in line_colors}
    return "_".join(c for c in color_order if c in found)


def map_comma_float(text) -> float:
    """'12 345,67' -> 12345.67. Strips regular, non-breaking and thin spaces."""
    if isinstance(text, (int, float)):
        return float(text)
    cleaned = re.sub(r'[\s\u00a0\u202f]', '', str(text)).replace(',', '.')
    return float(cleaned) if cleaned else float('nan')


def map_month_year(text: str) -> str:
    """'janv-2019' -> 'Jan-2019'. Checks 4-char keys first so juin != juil."""
    text_lower = remove_accents(str(text).lower())
    month, year = text_lower.split('-')
    month = FR_MONTHS.get(month[:4]) or FR_MONTHS.get(month[:3])
    if not (month and year):
        return ""
    return f"{EN_MONTHS[month]}-{'20'+year}"


def map_car_door(text) -> str:
    """Doors 1-8 in any combination, or 'toutes'. Open-ended, so not a dict.

    'toutes' -> 'all';  '3' -> '3';  '1;3' -> '1_3';  '2 et 5' -> '2_5'
    """
    text_lower = str(text).lower()
    if 'tout' in text_lower:
        return 'all'
    doors = sorted({int(n) for n in re.findall(r'\d+', text_lower) if 1 <= int(n) <= 8})
    return "_".join(str(d) for d in doors)


def map_evacuation(text) -> str:
    """Any combination of car / train / station, in canonical order.

    'Train et Station' -> 'train_station';  'Voiture' -> 'car'
    """
    text_lower = str(text).lower()
    parts = [('voiture', 'car'), ('train', 'train'), ('station', 'station')]
    return "_".join(en for fr, en in parts if fr in text_lower)


# def map_operating_period(text) -> str:
#     """Service periods are named for their starting month, in French.

#     'Janvier' -> 'January';  'P3-Juin 2021' -> 'P3-June 2021'
#     """
#     full = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
#             6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October',
#             11: 'November', 12: 'December'}
#     out = str(text)
#     for token in sorted(FR_MONTHS, key=len, reverse=True):
#         # consume the rest of the word too, so 'Janv' + 'ier' is fully replaced
#         match = re.search(token + r'[a-zéûàè]*', out, flags=re.IGNORECASE)
#         if match:
#             return out[:match.start()] + full[FR_MONTHS[token]] + out[match.end():]
#     return out


def map_day_type(text) -> str:
    """Weekday/weekend or an ordinal holiday period. Spelling varies in source
    ('Premiere', 'Premieme'), so match on stem rather than exact key."""
    t = str(text).lower()
    t = re.sub(r'[\u00e9\u00e8\u00ea]', 'e', t)
    if 'fete' in t or 'per.' in t:
        for stem, word in [('prem', 'first'), ('deux', 'second'),
                           ('trois', 'third'), ('quatr', 'fourth'),
                           ('cinq', 'fifth')]:
            if stem in t:
                return f'{word} holiday period'
        return 'holiday period'
    if 'semaine' in t:
        return 'weekday'
    if 'samedi' in t:
        return 'saturday'
    if 'dimanche' in t:
        return 'sunday'
    return ''


SCHEMA = {
    "incidents": [
        {"fr_name": "Numero d'incident", "en_name": "incident_number",
         "target_dtype": "string", "values": None},

        {"fr_name": "Type d'incident", "en_name": "type_of_incident",
         "target_dtype": "category",
         "values": {"S": "station", "T": "train"}},

        {"fr_name": "Ligne", "en_name": "line",
         "target_dtype": "category",
         "values": lambda x: map_line_colors(x)},

        {"fr_name": "Cause primaire", "en_name": "primary_cause",
         "target_dtype": "category",
         "values": {
             "Autres": "other",
             "Clientèle": "customers",
             "Équipements fixes": "fixed equipment",
             "Exploitation trains": "train operations",
             "Matériel roulant": "rolling stock",
         }},

        {"fr_name": "Cause secondaire", "en_name": "secondary_cause",
         "target_dtype": "category",
         "values": {
             "Autres": "other",
             "Causes externes": "external causes",
             "Méfait volontaire": "deliberate mischief",
             "Service aux trains": "train services department",
             "Nuisance involontaire": "unintentional nuisance",
             "Ligne 1, 2, 4, 5": "line infrastructure departments",
             "MR-73": "MR-73",
             "Blessée ou malade": "injured or ill person",
             "Service aux stations": "station services department",
             "MPM-10": "MPM-10",
             "Contrats Réno-Stations": "Reno-Stations contracts",
             "Contrats Réno-Système": "Reno-Systems contracts",
             "Service de la voie": "track department",
             "Pers. / Équipement STM": "STM personnel / equipment",
             "Centre de contrôle": "control centre",
             "Service TCPE": "TCPE department",
             "Contrats MPM10": "MPM10 contracts",
             "Véhicules de travaux": "work vehicles",
             "Pers. / Équipement Externe": "external personnel / equipment",
             "Matériel roulant": "rolling stock",
         }},

        {"fr_name": "Symptome", "en_name": "symptom",
         "target_dtype": "category",
         "values": {
             "Clientèle": "customers",
             "Exploitation stations": "station operations",
             "Feu, fumée, odeur, produit, etc...": "fire, smoke, odour, substance, etc.",
             "Équipements fixes": "fixed equipment",
             "Divers": "miscellaneous",
             "Exploitation trains": "train operations",
             "Matériel roulant": "rolling stock",
         }},

        {"fr_name": "Numéro de tournée", "en_name": "route",
         "target_dtype": "string", "values": lambda s: s.zfill(3) if s.isdigit() else s },

        {"fr_name": "Heure de l'incident", "en_name": "time_of_incident",
         "target_dtype": "service_time", "values": None},

        {"fr_name": "Heure de reprise", "en_name": "resumption_time",
         "target_dtype": "service_time", "values": None},

        {"fr_name": "Incident en minutes", "en_name": "incident_duration_band",
         "target_dtype": "ordered_category",
         "values": {
             "02 min et moins": "02 min and under",
             "03 à 04 min": "03 to 04 min",
             "05 à 09 min": "05 to 09 min",
             "10 à 14 min": "10 to 14 min",
             "15 à 19 min": "15 to 19 min",
             "20 à 24 min": "20 to 24 min",
             "25 à 29 min": "25 to 29 min",
             "30 min et plus": "30 min and over",
         }},

        {"fr_name": "Véhicule", "en_name": "vehicle",
         "target_dtype": "string", "values": None},

        {"fr_name": "Porte de voiture", "variants": ["Porte de voiture (numéro)"],
         "en_name": "car_door", "target_dtype": "category",
         "values": lambda x: map_car_door(x)},

        {"fr_name": "Type de matériel", "en_name": "hardware_type",
         "target_dtype": "category",
         "values": {"73": "MR-73", "10": "MPM-10"}},

        {"fr_name": "Code de lieu", "en_name": "location_code",
         "target_dtype": "string", "values": None},

        {"fr_name": "Dommage matériel", "en_name": "material_damage",
         "target_dtype": "boolean", "values": {"0": False, "1": True}},

        {"fr_name": "KFS", "en_name": "kfs",
         "target_dtype": "boolean", "values": {"0": False, "1": True}},

        {"fr_name": "Porte", "en_name": "door",
         "target_dtype": "boolean", "values": {"0": False, "1": True}},

        {"fr_name": "Urgence métro", "en_name": "emergency_metro",
         "target_dtype": "boolean", "values": {"0": False, "1": True}},

        {"fr_name": "CAT", "en_name": "cat",
         "target_dtype": "boolean", "values": {"0": False, "1": True}},

        {"fr_name": "Évacuation", "en_name": "evacuation",
         "target_dtype": "category",
         "values": lambda x: map_evacuation(x)},

        {"fr_name": "Année civile", "en_name": "calendar_year",
         "target_dtype": "Int64", "values": None},

        {"fr_name": "Année civile/mois", "en_name": "calendar_year_month",
         "target_dtype": "month_fr",
         "values": lambda x: map_month_year(x)},

        {"fr_name": "Mois calendrier", "en_name": "calendar_month",
         "target_dtype": "Int64", "values": None},

        {"fr_name": "Jour du mois", "en_name": "day_of_the_month",
         "target_dtype": "Int64", "values": None},

        {"fr_name": "Jour de la semaine", "en_name": "day_of_the_week",
         "target_dtype": "ordered_category",
         "values": {"1": "monday", "2": "tuesday", "3": "wednesday",
                    "4": "thursday", "5": "friday", "6": "saturday",
                    "7": "sunday"}},

        {"fr_name": "Jour calendaire", "en_name": "calendar_day",
         "target_dtype": "date", "values": None},
    ],

    "mileage": [
        {"fr_name": "Ligne", "en_name": "line",
         "target_dtype": "category",
         "values": lambda x: map_line_colors(x)},

        {"fr_name": "Période Exploitation", "en_name": "operating_period",
         "target_dtype": "string",
         "values": {}},

        {"fr_name": "Type de Jour", "en_name": "day_type",
         "target_dtype": "category",
         "values": lambda x: map_day_type(x)},

        {"fr_name": "Tournée", "variants": ["Tournee"], "en_name": "route",
         "target_dtype": "string", "values": None},

        {"fr_name": "KM planifié", "en_name": "planned_mileage",
         "target_dtype": "float",
         "values": lambda x: map_comma_float(x)},

        {"fr_name": "Année civile", "en_name": "calendar_year",
         "target_dtype": "Int64", "values": None},

        {"fr_name": "Année civile/mois", "en_name": "calendar_year_month",
         "target_dtype": "month_iso", "values": None},

        {"fr_name": "Année civile/semaine", "en_name": "calendar_week_start",
         "target_dtype": "date", "values": None},

        {"fr_name": "Jour calendaire", "en_name": "calendar_day",
         "target_dtype": "date", "values": None},

        {"fr_name": "Jour de la semaine", "en_name": "day_of_the_week",
         "target_dtype": "ordered_category",
         "values": {"Lundi": "monday", "Mardi": "tuesday",
                    "Mercredi": "wednesday", "Jeudi": "thursday",
                    "Vendredi": "friday", "Samedi": "saturday",
                    "Dimanche": "sunday"}},

        {"fr_name": "Mois calendrier", "en_name": "calendar_month",
         "target_dtype": "Int64", "values": None},
    ],
}

COL_VOCAB_MAP = {
    'incidents': {
        'Année civile': 'calendar_year',
        'Année civile/mois': 'calendar_year/month',
        'CAT': 'cat',
        'Cause primaire': 'primary_cause',
        'Cause secondaire': 'secondary_cause',
        'Code de lieu': 'location_code',
        'Dommage matériel': 'material_damage',
        "Heure de l'incident": 'time_of_incident',
        'Heure de reprise': 'resumption_time',
        'Incident en minutes': 'incident_in_minutes',
        'Jour calendaire': 'calendar_day',
        'Jour de la semaine': 'day_of_the_week',
        'Jour du mois': 'day_of_the_month',
        'KFS': 'kfs',
        'Ligne': 'line',
        'Mois calendrier': 'calendar_month',
        "Numero d'incident": 'incident_number',
        'Numéro de tournée': 'tour_number',
        'Porte': 'door',
        'Porte de voiture': 'car_door',
        'Symptome': 'symptom',
        "Type d'incident": 'type_of_incident',
        'Type de matériel': 'hardware_type',
        'Urgence métro': 'emergency_metro',
        'Véhicule': 'vehicle',
        'Évacuation': 'evacuation'
    },
    
    'mileage': {
        'Année civile': 'calendar_year',
        'Année civile/mois': 'calendar_year/month',
        'Année civile/semaine': 'calendar_year/week',
        'Jour calendaire': 'calendar_day',
        'Jour de la semaine': 'day_of_the_week',
        'KM planifié': 'planned_mileage',
        'Ligne': 'route',
        'Mois calendrier': 'calendar_month',
        'Période Exploitation': 'operating_period',
        'Tournée': 'route',
        'Type de Jour': 'day_type'
    }
}
