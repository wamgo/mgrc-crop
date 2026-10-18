"""Version-2 experiments for the JSTARS manuscript: phenology-based crop groups, four classifiers, strict error levels.

Writes results/exp/v2/<exp>_<mode>__<dataset>.parquet and never touches earlier result files.
The dataset must match MGRC_DATASET (the taxonomy is loaded at import time in experiments.py):

    python run_v2.py breizh main pac              (MGRC_DATASET unset or breizh)
    MGRC_DATASET=eurocrops python run_v2.py eurocrops main pac

Experiments: main, rho, year (BreizhCrops only), spatial, noise. Mode: pac or crc.
"""
import itertools, json, os, sys, time, concurrent.futures
import pandas as pd
import experiments as E
import noise_exp as N

OUT = E.OUT / ('v2' if os.environ.get('MGRC_TAXON', 'pheno') == 'pheno' else 'v2_register')
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['mlp', 'lgbm', 'tempcnn', 'ltae']
REG = ['frh01', 'frh02', 'frh03', 'frh04']
B5 = [25, 50, 100, 200, 400]
SPEC = {
    'main': (B5, [10], [0.02, 0.05, 0.1, 0.2], 50, 'random'),
    'rho': ([25, 50, 100, 200], [1, 2, 5, 20], [0.02, 0.1], 50, 'random'),
    'year': (B5, [10], [0.02, 0.05, 0.1], 50, 'random'),
    'spatial': ([50, 100, 400], [10], [0.02, 0.1], 50, 'spatial'),
    'register': (B5, [10], [0.02, 0.1], 50, 'random'),
}
SEED0 = {'breizh': 50000, 'eurocrops': 60000}


def pairs(dataset, exp):
    if dataset == 'breizh':
        if exp == 'year':
            return [(f'L1C2017_{s}', m, t) for s in ['frh01', 'frh04'] for m in MODELS
                    for t in ['2018_L1C_frh01', '2018_L1C_frh04', '2017_L1C_' + ('frh04' if s == 'frh01' else 'frh01')]]
        return [(f'L2A2017_{s}', m, f'2017_L2A_{t}') for s, t in itertools.product(REG, REG) if s != t for m in MODELS]
    doms = json.loads((E.ROOT / 'data' / 'eurocrops_domains.json').read_text(encoding='utf8'))
    out = []
    for s in ['LV', 'EE']:
        o = 'EE' if s == 'LV' else 'LV'
        for t in sorted(doms):
            if t.startswith(o) and len(t) > 2:
                out += [(f'EC2021_{s}', m, f'2021_EC_{t}') for m in MODELS]
    return out


def main(dataset, exp, mode, workers=12):
    assert E.DATASET == dataset, f'set MGRC_DATASET={dataset}'
    out = OUT / f'{exp}_{mode}__{dataset}.parquet'
    if out.exists():
        print('exists', out.name, flush=True)
        return
    P = pairs(dataset, exp)
    # one checkpoint file per pair, so an interrupted run resumes where it stopped
    ck = OUT / 'ck' / f'{exp}_{mode}__{dataset}'
    ck.mkdir(parents=True, exist_ok=True)
    todo = [(i, p) for i, p in enumerate(P) if not (ck / f'{i:03d}.parquet').exists()]
    print('pairs', len(P), 'to run', len(todo), flush=True)
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        if exp == 'noise':
            futs = {ex.submit(N.run, *p, [0.005, 0.01, 0.02], ['symmetric', 'confusion', 'adversarial'], reps=30,
                              B=100, alpha=0.02, rho=10, seed=SEED0[dataset] + 900 + i): i for i, p in todo}
        else:
            futs = {ex.submit(E.run_pair, *p, *SPEC[exp], mode=mode, seed=SEED0[dataset] + i): i for i, p in todo}
        for j, f in enumerate(concurrent.futures.as_completed(futs)):
            pd.DataFrame(f.result()).to_parquet(ck / f'{futs[f]:03d}.parquet')
            print('DONE', exp, mode, dataset, j + 1, len(todo), round(time.time() - t0), flush=True)
    df = pd.concat([pd.read_parquet(ck / f'{i:03d}.parquet') for i in range(len(P))], ignore_index=True)
    df.to_parquet(out)
    print('SAVED', out.name, len(df), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
