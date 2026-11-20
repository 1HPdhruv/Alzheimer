"""Publication boundary checks; file review is still required."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {'AUDIT', 'docs', 'configs', 'src', 'experiments', 'tests', 'scripts', 'notebooks', 'results'}
ALLOWED_TOP = {'README.md', 'LICENSE', 'CITATION.cff', 'CONTRIBUTING.md', 'requirements.txt', 'pyproject.toml', '.gitignore', 'PUBLICATION_AUDIT.md'}
ALLOWED_CSV = {'FILE_CLASSIFICATION.csv', 'SOURCE_MANIFEST.csv', 'LARGE_FILES.csv', 'DUPLICATE_REPORT.csv', 'RELEASE_FILES.csv', 'SCIENTIFIC_COMPONENTS.csv'}
BAD_PARTS = {'data', 'local_outputs', 'PRIVATE_participant_level', '.git', '.venv', 'node_modules', '__pycache__'}
SECRETS = [r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}', r'AKIA[0-9A-Z]{16}', r'sk-[A-Za-z0-9_-]{24,}', r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', r'(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*["\x27][A-Za-z0-9_+/-]{16,}["\x27]']
PRIVATE_PATH = re.compile(r'(?:/' + r'Users/|/' + r'home/)[^\s/]+')
IDENTIFIER = re.compile(r'\b(?:\d{3}_S_\d{4,6}|NACC\d{4,})\b')
SYNTHETIC_ID_FILES = {'tests/synthetic_baseline.py', 'tests/test_baseline_library.py'}

def inspect_bytes(name: str, data: bytes) -> list[str]:
    p = Path(name); problems = []
    if p.is_absolute() or '..' in p.parts:
        return ['unsafe relative path']
    if (len(p.parts) == 1 and name not in ALLOWED_TOP) or (len(p.parts) > 1 and p.parts[0] not in ALLOWED_ROOTS):
        problems.append('outside release allowlist')
    parts = list(p.parts)
    if parts[:3] == ['src', 'yoad', 'data']:
        parts = parts[3:]
    if any(x in BAD_PARTS for x in parts):
        problems.append('restricted directory')
    if p.suffix not in {'.py', '.md', '.yaml', '.toml', '.txt', '.cff', '.csv', ''} and name != '.gitignore':
        problems.append('unapproved file type')
    if p.suffix == '.csv' and (p.parts[0] != 'AUDIT' or p.name not in ALLOWED_CSV):
        problems.append('unapproved table')
    if len(data) > 5 * 1024 * 1024:
        problems.append('over 5 MiB release limit')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError:
        return problems + ['binary/non-UTF-8 content']
    if any(re.search(pattern, text) for pattern in SECRETS):
        problems.append('potential credential')
    if PRIVATE_PATH.search(text):
        problems.append('private absolute path')
    if IDENTIFIER.search(text) and name not in SYNTHETIC_ID_FILES:
        problems.append('potential participant identifier')
    return problems

def scan(staged: bool = False) -> dict:
    cmd = ['git', '-C', str(ROOT), 'ls-files', '-z', '--cached']
    if not staged:
        cmd += ['--others', '--exclude-standard']
    names = sorted(set(filter(None, subprocess.check_output(cmd).decode().split('\0'))))
    findings = []
    for name in names:
        p = ROOT / name
        if staged:
            mode = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-s', '--', name], text=True).split()[0]
            if mode != '100644':
                findings.append({'path': name, 'issues': ['unapproved staged mode']}); continue
            data = subprocess.check_output(['git', '-C', str(ROOT), 'show', ':' + name])
        else:
            if p.is_symlink():
                findings.append({'path': name, 'issues': ['symlink']}); continue
            data = p.read_bytes()
        issues = inspect_bytes(name, data)
        if issues:
            findings.append({'path': name, 'issues': issues})
    return {'status': 'PASS' if not findings and names else 'FAIL', 'files_checked': len(names), 'staged': staged, 'findings': findings}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--staged', action='store_true')
    result = scan(parser.parse_args().staged)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')
