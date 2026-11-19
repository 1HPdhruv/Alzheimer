"""Release boundary tests, using invented strings rather than real credentials/data."""
from scripts.publication_scan import inspect_bytes

def test_rejects_binary_and_unapproved_dataset():
    assert 'binary/non-UTF-8 content' in inspect_bytes('data/cohort.npz', b'\xff')
    assert 'unapproved table' in inspect_bytes('results/tables/cohort.csv', b'id,label\n')

def test_detects_secret_and_private_path_without_echoing_value():
    fake = ('gh' + 'p_' + 'X' * 30).encode()
    assert inspect_bytes('docs/example.md', fake) == ['potential credential']
    path = ('/' + 'Users' + '/example/private.csv').encode()
    assert 'private absolute path' in inspect_bytes('docs/example.md', path)

def test_detects_identifier_outside_invented_fixture():
    fake = ('123' + '_S_' + '00001').encode()
    assert inspect_bytes('results/summary.md', fake) == ['potential participant identifier']
    assert inspect_bytes('tests/synthetic_baseline.py', fake) == []

def test_allows_source_package_and_rejects_oversized_content():
    assert inspect_bytes('src/yoad/data/example.py', b'"""Pure functions."""\n') == []
    assert 'over 5 MiB release limit' in inspect_bytes('docs/example.md', b'x' * (5 * 1024 * 1024 + 1))
