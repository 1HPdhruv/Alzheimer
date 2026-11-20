"""Organization and publication boundaries; no participant-level execution."""
from pathlib import Path
import csv
import re

from scripts.publication_scan import inspect_bytes, SUBMISSION_INVENTORY

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'YOAD_SUBMISSION'
EXPECTED = {
    '01_LITERATURE_REVIEW': ['Literature_Review', 'Research_Gap', 'Research_Question_and_Hypotheses', 'Research_Methodology'],
    '02_EXPERIMENTATION': ['Dataset_and_Cohort_Audit', 'Source_Code', 'Experiment_Configurations', 'Execution_Logs', 'Model_Implementations'],
    '03_ANALYSIS': ['Model_Comparisons', 'Calibration_Analysis', 'Statistical_Analysis', 'Figures', 'Result_Tables', 'Robustness_Analysis'],
    '04_PAPER_DRAFT': ['Manuscript', 'Architecture_Diagrams', 'Figures', 'References', 'Supplementary_Material'],
    '05_SUBMISSION': ['Final_Manuscript', 'Cover_Letter', 'Author_Declarations', 'Supplementary_Files', 'Submission_Confirmation'],
    '06_ACCEPTANCE': ['Acceptance_Letter', 'Reviewer_Comments', 'Reviewer_Responses', 'Camera_Ready_Paper', 'Copyright_Forms', 'Publication_DOI'],
}

def inventory():
    with (PACKAGE / 'FILE_INVENTORY.csv').open() as handle:
        return list(csv.DictReader(handle))

def test_complete_requested_structure_has_meaningful_guides():
    assert (PACKAGE / 'README.md').is_file()
    for milestone, leaves in EXPECTED.items():
        assert (PACKAGE / milestone / 'README.md').stat().st_size > 300
        assert {p.name for p in (PACKAGE / milestone).iterdir() if p.is_dir()} == set(leaves)
        for leaf in leaves:
            text = (PACKAGE / milestone / leaf / 'README.md').read_text()
            assert 'Purpose:' in text and 'Currently available:' in text and 'Before progression:' in text
    assert not list(PACKAGE.rglob('*.docx')) and not list(PACKAGE.rglob('*.pdf'))

def test_all_copies_are_traceable_and_unchanged():
    copies = [row for row in inventory() if row['action_taken'] == 'COPIED']
    assert len(copies) == 11
    for row in copies:
        assert (ROOT / row['original_path']).read_bytes() == (ROOT / row['destination_path']).read_bytes()

def test_inventory_destinations_exist_and_are_relative():
    for row in inventory():
        for destination in filter(None, row['destination_path'].split('; ')):
            assert not Path(destination).is_absolute() and '..' not in Path(destination).parts
            assert (ROOT / destination).is_file(), destination
        if row['action_taken'] == 'EXCLUDED':
            assert row['original_path'].startswith('workspace-private:')
            assert not row['destination_path']

def test_no_uninventoried_package_files_or_private_payloads():
    known = {row['destination_path'] for row in inventory() if row['action_taken'] in {'GENERATED', 'COPIED'}}
    for path in PACKAGE.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():
            name = path.relative_to(ROOT).as_posix()
            assert name in known
            assert inspect_bytes(name, path.read_bytes()) == [], name

def test_tracker_counts_match_actual_rows_and_pending_evidence_is_explicit():
    text = (PACKAGE / 'SUBMISSION_MASTER_TRACKER.md').read_text()
    rows = [line.split('|')[1:-1] for line in text.splitlines() if line.startswith('| 0')]
    assert len(rows) == 48
    statuses = [row[4].strip() for row in rows]
    assert statuses.count('READY') == 15
    assert statuses.count('EXISTS — REVIEW REQUIRED') == 16
    assert statuses.count('PARTIALLY COMPLETE') == 4
    assert statuses.count('BLOCKED') == 4
    assert statuses.count('PENDING') == 9
    assert all(row[4].strip() == 'PENDING' for row in rows if row[0].strip() == '06_ACCEPTANCE')
    assert 'not completion measurements' in (PACKAGE / 'README.md').read_text()

def test_new_paths_do_not_allow_arbitrary_submission_payloads():
    assert inspect_bytes('YOAD_SUBMISSION/README.md', b'Organization guide.') == []
    assert 'outside release allowlist' in inspect_bytes('YOAD_SUBMISSION/extra.md', b'Text')
    assert 'unapproved table' in inspect_bytes('YOAD_SUBMISSION/results.csv', b'label,prediction\n')
    assert 'unapproved file type' in inspect_bytes('YOAD_SUBMISSION/05_SUBMISSION/Final_Manuscript/final.pdf', b'%PDF')
    assert 'invalid submission inventory schema' in inspect_bytes(SUBMISSION_INVENTORY, b'label,score\n0,0\n')

def test_existing_secret_privacy_and_size_guards_apply_to_new_paths():
    name = 'YOAD_SUBMISSION/README.md'
    assert 'potential credential' in inspect_bytes(name, ('gh' + 'p_' + 'Z' * 30).encode())
    assert 'private absolute path' in inspect_bytes(name, ('/' + 'Users' + '/example/private').encode())
    assert 'potential participant identifier' in inspect_bytes(name, ('123' + '_S_' + '00001').encode())
    assert 'over 5 MiB release limit' in inspect_bytes(name, b'x' * (5 * 1024 * 1024 + 1))
    assert 'binary/non-UTF-8 content' in inspect_bytes(name, b'\xff')

def test_all_relative_markdown_links_resolve():
    for path in PACKAGE.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in target and not target.startswith('#'):
                assert (path.parent / target.split('#')[0]).exists(), (path, target)
