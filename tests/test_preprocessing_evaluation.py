import pandas as pd, numpy as np, pytest
from yoad.preprocessing.align import *
from yoad.evaluation.safeguards import *
from yoad.reporting.report import write_report

def test_align_nearest_window_and_unmatched():
    base = pd.DataFrame({"participant_id":["a","b","c"],"baseline_date":pd.to_datetime(["2020-01-01"]*3)})
    mri = pd.DataFrame({"participant_id":["a","a","b"],"visit_date":pd.to_datetime(["2020-01-20","2019-12-30","2020-06-01"])})
    r = align_nearest(base, mri, 90).set_index("participant_id")
    assert r.delta_days["a"]==-2 and not r.matched["b"] and not r.matched["c"]

def test_missingness_and_overlap():
    m = missingness_summary(pd.DataFrame({"x":[1,None],"y":[1,2]})).set_index("column"); assert m.n_missing["x"]==1
    o = modality_overlap({"mri":{1,2},"cog":{2,3}}).set_index("pattern").n
    assert o["mri+cog"]==1 and o["mri"]==1 and o["cog"]==1

def test_visit_counts(long_df): assert visit_counts(long_df).set_index("stat").value["max"]==3

def test_feature_leakage_names():
    hits = check_feature_names(["age","time_to_event","MMSE","converted_flag","y"], ["y"])
    assert set(hits)=={"time_to_event","converted_flag","y"}

def test_feature_dates_postbaseline():
    base = pd.DataFrame({"participant_id":[1],"baseline_date":pd.to_datetime(["2020-01-01"])})
    f = pd.DataFrame({"participant_id":[1,1],"visit_date":pd.to_datetime(["2019-12-01","2020-03-01"])})
    assert len(check_feature_dates(f, base))==1

def test_group_kfold_no_participant_overlap():
    ids = pd.Series(np.repeat(np.arange(30),3)); y = pd.Series(np.repeat(np.arange(30)%3==0,3).astype(int))
    folds = group_kfold(ids, y, k=5, seed=1)
    for tr,te in folds: assert not set(ids.iloc[tr])&set(ids.iloc[te])
    assert sorted(np.concatenate([te for _,te in folds]))==list(range(90))

def test_assert_disjoint_raises():
    with pytest.raises(LeakageError): assert_disjoint([1,2],[2,3])

def test_report_writes(tmp_path):
    write_report({"a":{"x":1,"y":[1,2]}}, tmp_path/"r.md", tmp_path/"r.json"); assert "x" in (tmp_path/"r.md").read_text()
