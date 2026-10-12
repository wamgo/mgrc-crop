import itertools, time, concurrent.futures
import pandas as pd
import noise_exp as N
if __name__ == '__main__':
    regions = ['frh01', 'frh02', 'frh03', 'frh04']
    pairs = [(f'L2A2017_{s}', m, f'2017_L2A_{t}') for s, t, m in itertools.product(regions, regions, ['mlp', 'lgbm', 'tempcnn']) if s != t]
    rows = []
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(N.run, *p, [0.01, 0.02, 0.05], ['symmetric', 'confusion', 'adversarial'], reps=30, B=50, seed=500 + i) for i, p in enumerate(pairs)]
        for i, f in enumerate(concurrent.futures.as_completed(futs)):
            rows += f.result(); print('DONE', i + 1, len(pairs), round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_parquet(N.OUT / 'noise_b50_pac.parquet'); print('SAVED')
