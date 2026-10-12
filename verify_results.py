"""Recompute manuscript headline statistics from the released per-run outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
KEYS = ['tag', 'model', 'target', 'B', 'rho', 'alpha']


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        data = (ROOT / name).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise AssertionError(f'Release checksum mismatch: {name}')
    report = {}
    for name, rows, settings, expected in [('main_pac', 99000, 36, .8355555555555556),
                                            ('year_pac', 49500, 18, .6166666666666667)]:
        df = pd.read_parquet(ROOT / 'results' / 'exp' / f'{name}.parquet')
        assert len(df) == rows
        assert len(df[['tag', 'model', 'target']].drop_duplicates()) == settings
        for column in ['risk', 'fine', 'parent', 'root', 'hcov']:
            assert np.isfinite(df[column]).all(), column
            assert df[column].between(-1e-12, 1 + 1e-12).all(), column
        assert np.allclose(df.fine + df.parent + df.root, 1)
        source = df[df.method == 'source_levelwise']
        violation = float((source.risk > source.alpha).mean())
        assert np.isclose(violation, expected, atol=1e-12, rtol=0)
        report[name] = {'rows': rows, 'settings': settings, 'source_violation': violation}
        if name == 'main_pac':
            means = df.groupby(KEYS + ['method']).hcov.mean().unstack('method')
            best = means[['flat_selective', 'hsc_climb', 'levelwise_path_fine']].max(axis=1)
            gains = (means.planned_info - best).groupby('B').mean()
            assert np.allclose(gains.loc[[25, 50, 100]], [.09326, .08742, .03234], atol=5e-6, rtol=0)
            report[name]['coverage_gain_vs_best_fine'] = gains.to_dict()
        else:
            settings_df = df[['tag', 'model', 'target']].drop_duplicates()
            assert settings_df.target.str.startswith('2018_').sum() == 12
            assert settings_df.target.str.startswith('2017_').sum() == 6
    for filename in ['noise_pac', 'noise_b50_pac']:
        df = pd.read_parquet(ROOT / 'results' / 'exp' / f'{filename}.parquet')
        corrected = df[df.method == 'mgrc_corrected_info'].assign(over=lambda x: x.risk > .1)
        maximum = float(corrected.groupby(['kind', 'eta']).over.mean().max())
        assert maximum <= .064
        report[filename] = {'rows': len(df), 'max_corrected_aggregate_violation': maximum}
    print(json.dumps(report, indent=2))
    print('Release checksums and headline-result checks passed.')


if __name__ == '__main__':
    main()
