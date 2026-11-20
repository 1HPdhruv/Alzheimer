"""Synthetic fixture extracted from historical software tests; no actual participant records."""
import numpy as np
import pandas as pd
MRI = ["ST1SV", "ST10CV", "ST10SA", "ST10TA", "ST10TS", "ST2SV"]

def raw_tables():
    """Hand-built ADNI-like source tables (column names as in the reference code)."""
    dx = [  # PTID, RID, VISCODE2, EXAMDATE, DIAGNOSIS (1 CN, 2 MCI, 3 DEM)
        ("999_S_0001", 1, "bl", "2010-01-10", 2), ("999_S_0001", 1, "m06", "2010-07-10", 2), ("999_S_0001", 1, "m12", None, 3),  # undated DEM
        ("999_S_0002", 2, "bl", "2011-03-01", 2), ("999_S_0002", 2, "m06", "2011-09-01", 2), ("999_S_0002", 2, "m12", "2012-03-01", 2),
        ("999_S_0002", 2, "m18", "2012-09-01", 2), ("999_S_0002", 2, "m24", "2013-03-01", 2),
        ("999_S_0003", 3, "bl", "2012-01-01", 2), ("999_S_0003", 3, "m06", "2012-07-01", 1), ("999_S_0003", 3, "m24", "2014-01-01", 2),  # CN -> excluded
        ("381_S_99999", 4, "bl", "2012-01-01", 2), ("381_S_99999", 4, "m24", "2014-01-01", 3),                                          # advisory
    ]
    dx = pd.DataFrame(dx, columns=["PTID", "RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS"])
    vis = [(1, "bl"), (1, "m06"), (2, "bl"), (2, "m06"), (2, "m12"), (2, "m18"), (2, "m24"), (3, "bl"), (4, "bl")]
    mm = pd.DataFrame([(r, v, 28.0, None) for r, v in vis], columns=["RID", "VISCODE2", "MMSCORE", "VISDATE"])
    mm.loc[(mm.RID == 1) & (mm.VISCODE2 == "m06"), "MMSCORE"] = -4.0                      # ADNI missing code
    mm = pd.concat([mm, pd.DataFrame([(2, "bl", np.nan, None)], columns=mm.columns)])     # duplicate, more missing -> dropped
    cdr = pd.DataFrame([(r, v, 0.5, 1.5) for r, v in vis], columns=["RID", "VISCODE2", "CDGLOBAL", "CDRSB"])
    adas = pd.DataFrame([(r, v, 10.0, 16.0) for r, v in vis], columns=["RID", "VISCODE2", "TOTSCORE", "TOTAL13"])
    faq = pd.DataFrame([(r, v, 3.0) for r, v in vis], columns=["RID", "VISCODE2", "FAQTOTAL"])
    ap = pd.DataFrame({"RID": [1, 2, 3, 4], "GENOTYPE": ["3/4", "3/3", "4/4", "3/3"]})
    dm = pd.DataFrame({"RID": [1, 2, 3, 4], "PTGENDER": [2, 1, 1, 2], "PTDOB": ["1940-01-10", "1950-03-01", "1945-01-01", "1950-01-01"],
                       "PTHAND": [1, 2, 1, 1], "PTEDUCAT": [16, 12, 14, 18]})
    fs = pd.DataFrame([(r, v, *np.arange(len(MRI)) + 1.0) for r, v in vis], columns=["RID", "VISCODE2", *MRI])
    fs["OTHERCOL"] = 1.0
    return {"dxsum": dx, "mmse": mm, "cdr": cdr, "adas": adas, "faq": faq, "apoeres": ap, "ptdemog": dm, "ucsffsx7": fs}
