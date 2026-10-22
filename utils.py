import unicodedata
from datetime import date

import numpy as np


def norm(s) -> str:
    """Lowercase and strip accents/extra spaces so 'Ødegaard' ~ 'odegaard'."""
    s = unicodedata.normalize("NFKD", str(s))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).lower().split())


def age_from_dob(dob, today=None) -> float:
    """Age in years (1 d.p.) from an ISO date string; NaN if it can't be parsed."""
    today = today or date.today()
    try:
        y, m, d = map(int, str(dob)[:10].split("-"))
        return round((today - date(y, m, d)).days / 365.25, 1)
    except Exception:
        return np.nan
