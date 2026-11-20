import pandas as pd, pytest

def d(s): return pd.Timestamp(s)

@pytest.fixture
def long_df():
    """Synthetic. P1 MCI->AD at 1y; P2 MCI stable 3y; P3 MCI, 6 months follow-up only; P4 MCI->other dementia 1y;
    P5 MCI->AD then reverts to MCI; P6 no follow-up; P7 CN baseline only; P8 same-day conflicting dx."""
    rows = [
        ("P1","2020-01-01","MCI"),("P1","2021-01-01","AD"),
        ("P2","2020-01-01","MCI"),("P2","2021-01-01","MCI"),("P2","2023-01-01","MCI"),
        ("P3","2020-01-01","MCI"),("P3","2020-07-01","MCI"),
        ("P4","2020-01-01","MCI"),("P4","2021-01-01","OTHER_DEM"),
        ("P5","2020-01-01","MCI"),("P5","2020-12-01","AD"),("P5","2021-12-01","MCI"),
        ("P6","2020-01-01","MCI"),
        ("P7","2020-01-01","CN"),("P7","2021-01-01","CN"),
        ("P8","2020-01-01","MCI"),("P8","2020-01-01","AD"),
        ("P9",None,"MCI"),
    ]
    return pd.DataFrame(rows, columns=["participant_id","visit_date","diagnosis"]).assign(visit_date=lambda x: pd.to_datetime(x.visit_date))
