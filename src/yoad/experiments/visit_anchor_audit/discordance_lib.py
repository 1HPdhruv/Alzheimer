"""Exploratory audit v1: paired labels and ascertainment, without model fitting.

Pure in-memory functions. Real inputs are accessible only through the guarded
run_discordance_audit.py entry point. Frozen cohort/label functions are reused.
Synthetic fixtures test software only; no row-level output is returned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yoad.experiments.visit_anchor_audit import design_lib as frozen
from yoad.experiments.visit_anchor_audit.audit_lib import adsl_death_dates
from yoad.features.anchor_predictors import attach_predictors, eligible_anchors

VERSION = 'label-discordance-ascertainment-v1'
KEY = ['participant_id', 'anchor_month']
STATES = ('EVENT', 'NONEVENT', 'NOT_LABELLED')
CATEGORIES = ('event_both', 'nonevent_both', 'SIM_event_REF_nonevent', 'SIM_nonevent_REF_event',
              'SIM_only_labelled', 'REF_only_labelled', 'neither_labelled')
DETERMINATE = ('EVENT', 'NONEVENT')


def normalise(s):
    if not s.isin(['EVENT', 'NONEVENT', 'INDETERMINATE', 'EXCLUDED']).all():
        raise ValueError('Unexpected outcome state')
    return s.where(s.isin(DETERMINATE), 'NOT_LABELLED')


def categories(a, b):
    a, b = normalise(a), normalise(b)
    return pd.Series(np.select([
        (a == 'EVENT') & (b == 'EVENT'), (a == 'NONEVENT') & (b == 'NONEVENT'),
        (a == 'EVENT') & (b == 'NONEVENT'), (a == 'NONEVENT') & (b == 'EVENT'),
        a.isin(DETERMINATE) & (b == 'NOT_LABELLED'),
        (a == 'NOT_LABELLED') & b.isin(DETERMINATE),
    ], CATEGORIES[:-1], default=CATEGORIES[-1]), index=a.index)


def public_count(n):
    n = int(n)
    return n if n == 0 or n >= 11 else '<11'


def public_signed(n):
    return int(n) if n == 0 or abs(n) >= 11 else 'suppressed (absolute value <11)'


def proportion(n, d):
    return float(n / d) if d >= 11 and (n == 0 or n >= 11) else None


def measure(row, mask, denominator):
    """Counts of anchors and their unique participants, with distinct denominators."""
    mask = pd.Series(mask, index=denominator.index).astype(bool)
    chosen = denominator.loc[mask]
    n, d = len(chosen), len(denominator)
    npart, dpart = chosen.participant_id.nunique(), denominator.participant_id.nunique()
    return {**row, 'n_anchors': public_count(n), 'anchor_denominator': public_count(d),
            'anchor_proportion': proportion(n, d), 'n_participants': public_count(npart),
            'participant_denominator': public_count(dpart), 'participant_proportion': proportion(npart, dpart)}


def validate_rows(rows):
    required = set(KEY + ['is_baseline', 'state_SIM', 'state_REF', 'reason_SIM', 'reason_REF', 'partition'])
    if required - set(rows):
        raise ValueError(f'Missing audit columns: {sorted(required - set(rows))}')
    if rows.participant_id.isna().any() or rows.duplicated(KEY).any():
        raise ValueError('Missing participant or duplicate participant-anchor')
    if not rows.partition.isin(['train', 'held_out']).all():
        raise ValueError('Invalid partition')
    if rows.groupby('participant_id').partition.nunique().max() != 1:
        raise ValueError('Participant appears in both partitions')
    if not rows.groupby('participant_id').is_baseline.sum().eq(1).all():
        raise ValueError('Every participant requires exactly one baseline anchor')
    normalise(rows.state_SIM); normalise(rows.state_REF)


def scopes(rows):
    for part in ('train', 'held_out', 'combined'):
        p = rows if part == 'combined' else rows[rows.partition == part]
        for anch in ('baseline', 'all_eligible_anchors'):
            yield dict(partition=part, anchoring=anch), p[p.is_baseline] if anch == 'baseline' else p


def paired_tables(rows):
    validate_rows(rows)
    cross, summary, selection, decomposition, reasons, strata = [], [], [], [], [], []
    for scope, r in scopes(rows):
        a, b = normalise(r.state_SIM), normalise(r.state_REF)
        cats = categories(r.state_SIM, r.state_REF)
        for sa in STATES:
            for sb in STATES:
                cross.append(measure({**scope, 'SIM_state': sa, 'REF_state': sb}, (a == sa) & (b == sb), r))
        for cat in CATEGORIES:
            summary.append(measure({**scope, 'category': cat}, cats == cat, r))
        common = a.isin(DETERMINATE) & b.isin(DETERMINATE)
        summary.append(measure({**scope, 'category': 'discordant_among_jointly_labelled'},
                               a[common] != b[common], r[common]))
        labelled_a, labelled_b = a.isin(DETERMINATE), b.isin(DETERMINATE)
        for unit in ('anchors', 'participants'):
            aset = set(map(tuple, r.loc[labelled_a, KEY].to_numpy())) if unit == 'anchors' else set(r.loc[labelled_a, 'participant_id'])
            bset = set(map(tuple, r.loc[labelled_b, KEY].to_numpy())) if unit == 'anchors' else set(r.loc[labelled_b, 'participant_id'])
            universe = set(map(tuple, r[KEY].to_numpy())) if unit == 'anchors' else set(r.participant_id)
            for label, ids in [('prelabel_eligible_both', universe), ('labelled_SIM', aset), ('labelled_REF', bset),
                               ('labelled_both', aset & bset), ('labelled_SIM_only', aset - bset),
                               ('labelled_REF_only', bset - aset), ('labelled_neither', universe - (aset | bset))]:
                selection.append({**scope, 'unit': unit, 'selection': label, 'count': public_count(len(ids)),
                                  'denominator': public_count(len(universe)), 'proportion': proportion(len(ids), len(universe))})
        terms = {
            'SIM_event_REF_nonevent': int(((a == 'EVENT') & (b == 'NONEVENT')).sum()),
            'SIM_nonevent_REF_event': -int(((a == 'NONEVENT') & (b == 'EVENT')).sum()),
            'SIM_event_REF_unlabelled': int(((a == 'EVENT') & (b == 'NOT_LABELLED')).sum()),
            'SIM_unlabelled_REF_event': -int(((a == 'NOT_LABELLED') & (b == 'EVENT')).sum()),
        }
        total = int((a == 'EVENT').sum() - (b == 'EVENT').sum())
        if sum(terms.values()) != total:
            raise AssertionError('Event-count decomposition failed')
        terms['net_common_label_disagreement'] = terms['SIM_event_REF_nonevent'] + terms['SIM_nonevent_REF_event']
        terms['net_differential_label_availability'] = terms['SIM_event_REF_unlabelled'] + terms['SIM_unlabelled_REF_event']
        terms['total_SIM_minus_REF_events'] = total
        for term, n in terms.items():
            decomposition.append({**scope, 'unit': 'anchors (participants at baseline only)', 'term': term,
                                  'signed_event_count': public_signed(n), 'anchor_denominator': public_count(len(r))})
        for rule in ('SIM', 'REF'):
            for (state, reason), group in r.groupby([f'state_{rule}', f'reason_{rule}'], dropna=False):
                reasons.append(measure({**scope, 'rule': rule, 'state': state, 'reason': reason}, r.index.isin(group.index), r))
        for column in ('followup_duration', 'm24_availability', 'ref_window_assessment', 'cognitive_availability'):
            if column not in r:
                continue
            for value, group in r.groupby(column, dropna=False):
                for cat in CATEGORIES:
                    strata.append(measure({**scope, 'stratifier': column, 'stratum': str(value), 'category': cat},
                                          cats.loc[group.index] == cat, group))
    return {name: pd.DataFrame(data) for name, data in [('label_crosstab', cross), ('discordance_summary', summary),
            ('selection_overlap', selection), ('event_decomposition', decomposition), ('ascertainment_reasons', reasons),
            ('ascertainment_strata', strata)]}


def trigger_mask(raw):
    return (raw.PHASE.astype(str).str.upper().eq('ADNI1') &
            pd.to_numeric(raw.DXAD, errors='coerce').eq(1) & pd.to_numeric(raw.DXOTHDEM, errors='coerce').eq(1))


def proposed_adni_dem_type_v1(raw):
    """Proposal only: simultaneous AD/OTHER ADNI1 row becomes UNKNOWN before visit consolidation.

    Retains frozen semantics for every other row. Never mutates the input or
    replaces the frozen mapper. UNKNOWN + AD rows remain UNKNOWN for that visit.
    """
    proposal = raw.copy()
    proposal.loc[trigger_mask(raw), ['DXAD', 'DXOTHDEM']] = np.nan
    return frozen.adni_dem_type(proposal)


def followup_features(rows, visits, cutoff, deaths):
    groups = {pid: g for pid, g in visits.groupby('participant_id')}
    records = []
    for a in rows.itertuples():
        v = groups[a.participant_id]
        after = v[(v.date > a.anchor_date) & (v.date <= pd.Timestamp(cutoff))]
        if a.participant_id in deaths:
            after = after[after.date <= deaths[a.participant_id]]
        lo, hi = a.horizon_date - pd.Timedelta(days=90), a.horizon_date + pd.Timedelta(days=90)
        last = after.date.max()
        duration = ('no_post_anchor_assessment' if pd.isna(last) else 'before_H_minus_90' if last < lo else
                    'H_minus_90_through_H_plus_90' if last <= hi else 'beyond_H_plus_90')
        nominal = v[v.visit_month.isin([a.anchor_month + k for k in (6, 12, 18, 24)])]
        m24 = nominal[nominal.visit_month == a.anchor_month + 24]
        window = after[after.date.between(lo, hi)]
        dem = after[after.dx == 'DEM']
        cognitive = sum(pd.notna(getattr(a, k)) for k in ('mmse', 'adas13', 'faq'))
        records.append(dict(followup_duration=duration, m24_availability='present' if len(m24) else 'absent',
                            ref_window_assessment='present' if len(window) else 'absent',
                            cognitive_availability='all_three' if cognitive == 3 else 'none' if cognitive == 0 else 'partial',
                            nominal_followup_after_cutoff=bool((nominal.date > pd.Timestamp(cutoff)).any()),
                            dementia_during_predictor_window=bool(dem.date.between(a.anchor_date + pd.Timedelta(days=1), a.index_time).any())))
    return pd.concat([rows.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def align_ref(original, corrected):
    if original.duplicated(KEY).any() or corrected.duplicated(KEY).any():
        raise ValueError('Duplicate keys in mapping comparison')
    c = corrected.set_index(KEY)
    o = original.set_index(KEY)
    if set(o.index) != set(c.index):
        raise ValueError('Mapping comparison changed anchor eligibility')
    c = c.reindex(o.index)
    if not o.state_SIM.equals(c.state_SIM):
        raise AssertionError('Etiology correction changed SIM')
    r = original.copy()
    r['corrected_state_REF'] = c.state_REF.to_numpy()
    r['corrected_reason_REF'] = c.reason_REF.to_numpy()
    return r


def mapping_tables(raw, visits, original, corrected):
    r = align_ref(original, corrected)
    old_map, new_map = frozen.adni_dem_type(raw), proposed_adni_dem_type_v1(raw)
    mapkeys = ['participant_id', 'visit_month']
    joined = old_map.merge(new_map, on=mapkeys, suffixes=('_old', '_proposed'), validate='one_to_one')
    retained = visits.merge(joined, on=mapkeys, how='left', validate='one_to_one')
    trigger = trigger_mask(raw)
    stages = []
    def stage(name, n, d, unit):
        stages.append(dict(stage=name, unit=unit, count=public_count(n), denominator=public_count(d), proportion=proportion(n, d)))
    stage('simultaneous_ADNI1_AD_and_OTHER_flags', int(trigger.sum()), len(raw), 'raw diagnosis rows after advisory exclusion')
    stage('trigger_rows_with_dementia_diagnosis', int((trigger & pd.to_numeric(raw.DIAGNOSIS, errors='coerce').eq(3)).sum()),
          len(raw), 'raw diagnosis rows after advisory exclusion')
    changed = retained.dem_type_old != retained.dem_type_proposed
    stage('retained_visits_with_changed_etiology', int(changed.sum()), len(retained), 'harmonised visits')
    stage('retained_dementia_visits_with_changed_etiology', int((changed & retained.dx.eq('DEM')).sum()), len(retained), 'harmonised visits')
    pop_ids = set(r.participant_id)
    inpop = retained.participant_id.isin(pop_ids)
    stage('analysis_participants_with_changed_dementia_etiology', retained.loc[changed & inpop & retained.dx.eq('DEM'), 'participant_id'].nunique(),
          len(pop_ids), 'unique analysis participants')
    transitions, impact = [], []
    for scope, t in scopes(r):
        for old in STATES:
            for new in STATES:
                mask = normalise(t.state_REF).eq(old) & normalise(t.corrected_state_REF).eq(new)
                transitions.append(measure({**scope, 'original_REF': old, 'proposed_REF': new}, mask, t))
        for label, mask in [('state_changed', t.state_REF != t.corrected_state_REF),
                            ('reason_changed', t.reason_REF != t.corrected_reason_REF),
                            ('event_to_not_labelled', t.state_REF.eq('EVENT') & ~t.corrected_state_REF.isin(DETERMINATE))]:
            impact.append(measure({**scope, 'impact': label}, mask, t))
    training = r.partition.eq('train')
    held_base = r.partition.eq('held_out') & r.is_baseline
    change = r.state_REF != r.corrected_state_REF
    # Booleans distinguish zero from positive impact without exposing small counts.
    decision = dict(any_eligible_label_changed=bool(change.any()),
                    training_baseline_label_changed=bool((change & training & r.is_baseline).any()),
                    training_any_anchor_label_changed=bool((change & training).any()),
                    heldout_baseline_target_changed=bool((change & held_base).any()),
                    heldout_any_anchor_target_changed=bool((change & ~training).any()))
    decision['recommendation'] = ('separately_versioned_corrected_AD_sensitivity_warranted' if decision['any_eligible_label_changed'] else
                                  'no_AD_sensitivity_rerun_needed_for_this_defect_on_these_inputs')
    return dict(mapping_stages=pd.DataFrame(stages), mapping_transitions=pd.DataFrame(transitions), mapping_impact=pd.DataFrame(impact)), decision


def build_audit(visits, raw, measurements, static, deaths, cutoff, seed=20261017, train_fraction=.6):
    anchors, flow = eligible_anchors(visits, cutoff, death_dates=deaths)
    rows, measurement_flow = attach_predictors(anchors, measurements, static, data_cutoff=cutoff)
    any_rows = frozen.label_anchors(visits, rows, cutoff, deaths, 'any')
    train, held = frozen.split_participants(any_rows, seed, train_fraction)
    any_rows['partition'] = np.where(any_rows.participant_id.isin(train), 'train', 'held_out')
    any_rows = followup_features(any_rows, visits, cutoff, deaths)
    original_visits = visits.merge(frozen.adni_dem_type(raw), on=['participant_id', 'visit_month'], how='left', validate='one_to_one')
    corrected_visits = visits.merge(proposed_adni_dem_type_v1(raw), on=['participant_id', 'visit_month'], how='left', validate='one_to_one')
    original = frozen.label_anchors(original_visits, rows, cutoff, deaths, 'ad')
    corrected = frozen.label_anchors(corrected_visits, rows, cutoff, deaths, 'ad')
    if frozen.split_participants(original, seed, train_fraction) != (train, held) or frozen.split_participants(corrected, seed, train_fraction) != (train, held):
        raise AssertionError('AD mapping comparison changed participant split')
    original['partition'] = any_rows.partition.to_numpy()
    corrected['partition'] = any_rows.partition.to_numpy()
    tables = paired_tables(any_rows)
    mapping, decision = mapping_tables(raw, visits, original, corrected)
    tables.update(mapping)
    tables['reconstructed_any_counts'] = frozen._counts(any_rows, train, held)
    tables['reconstructed_ad_counts'] = frozen._counts(original, train, held)
    tables['anchor_flow'] = pd.DataFrame([dict(step=k, count=public_count(v)) for k, v in {**flow, **measurement_flow}.items()])
    timing = []
    for scope, r in scopes(any_rows):
        for col in ('nominal_followup_after_cutoff', 'dementia_during_predictor_window'):
            timing.append(measure({**scope, 'check': col}, r[col], r))
    tables['timing_checks'] = pd.DataFrame(timing)
    return tables, dict(version=VERSION, mapping_decision=decision, inference='descriptive census; no confidence intervals; no model fitting')


def reconcile_counts(reconstructed, saved):
    keys = ['partition', 'rule', 'anchoring']
    a, b = reconstructed.sort_values(keys).reset_index(drop=True), saved.sort_values(keys).reset_index(drop=True)
    if set(a.columns) != set(b.columns) or not a.astype(str).equals(b[a.columns].astype(str)):
        raise ValueError('Reconstructed population/label margins do not match frozen run; do not report audit as corresponding')


def assert_aggregate_only(tables):
    forbidden = {'participant_id', 'RID', 'PTID', 'SUBJID', 'anchor_date', 'index_time', 'EXAMDATE', 'VISDATE'}
    for name, table in tables.items():
        if forbidden & set(table.columns):
            raise ValueError(f'Row-level field in aggregate output {name}')
