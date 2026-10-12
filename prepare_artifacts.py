"""Verify and unpack the reviewed results bundle without overwriting other data."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    bundle = ROOT / 'results_bundle.zip'
    if digest(bundle.read_bytes()) != manifest['files']['results_bundle.zip']['sha256']:
        raise ValueError('Bundle SHA-256 mismatch')
    with zipfile.ZipFile(bundle) as archive:
        expected = manifest['bundle_members']
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
            raise ValueError('Bundle file list differs from manifest')
        pending = []
        for name, entry in expected.items():
            destination = (ROOT / name).resolve()
            if not destination.is_relative_to(ROOT) or destination == ROOT:
                raise ValueError('Unsafe bundle path')
            data = archive.read(name)
            if len(data) != entry['bytes'] or digest(data) != entry['sha256']:
                raise ValueError(f'Bundle member checksum mismatch: {name}')
            if destination.exists():
                if not destination.is_file() or digest(destination.read_bytes()) != entry['sha256']:
                    raise FileExistsError(f'Refusing to replace different file: {name}')
            else:
                pending.append((destination, data))
        for destination, data in pending:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as target:
                target.write(data)
    print(f'Checked {len(expected)} bundle members; extracted {len(pending)} new files.')


if __name__ == '__main__':
    prepare()
