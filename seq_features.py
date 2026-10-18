"""Raw observation sequences for the attention-based classifier (L-TAE).

Parcels are taken in exactly the order of the existing feat_*.npz files, so the source train/dev/hold
split and the target domains are unchanged. Per parcel: up to T_MAX observations, cloudy ones included (10 bands,
reflectance/10000) with their day of year; if more are available, T_MAX are taken evenly in time.
"""
from pathlib import Path
import sys, io, zipfile
from pilot_common import setup
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd, h5py

D = ROOT / 'data'
T_MAX = 96
L1C_BANDS = [4, 5, 6, 7, 8, 9, 10, 11, 2, 3]  # B2..B8A, B11, B12 in the alphabetical L1C h5 layout
CSV_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']


def pack(bands, doy, valid):
    # All finite observations are kept: attention learns to down-weight cloudy dates, and in a pilot on
    # FRH01 this gave higher source accuracy (0.679) than clear-only sequences (0.665). `valid` is unused.
    valid = np.isfinite(bands).all(1)
    idx = np.flatnonzero(valid)
    if len(idx) > T_MAX:
        idx = idx[np.round(np.linspace(0, len(idx) - 1, T_MAX)).astype(int)]
    x = np.zeros((T_MAX, 10), np.float16)
    d = np.zeros(T_MAX, np.int16)
    m = np.zeros(T_MAX, bool)
    x[:len(idx)] = np.clip(bands[idx], 0, 2)
    d[:len(idx)] = doy[idx]
    m[:len(idx)] = True
    return x, d, m


def extract(year, level, region):
    out = D / f'seq_{year}_{level}_{region}.npz'
    if out.exists():
        return
    ids = np.load(D / f'feat_{year}_{level}_{region}.npz')['ids']
    X = np.zeros((len(ids), T_MAX, 10), np.float16)
    DOY = np.zeros((len(ids), T_MAX), np.int16)
    M = np.zeros((len(ids), T_MAX), bool)
    if year == 2018:
        zf = zipfile.ZipFile(D / '2018_L1C' / f'{region}.zip')
        names = {Path(n).stem: n for n in zf.namelist() if n.endswith('.csv')}
        for i, pid in enumerate(ids):
            t = pd.read_csv(io.BytesIO(zf.read(names[str(pid)])))
            doy = pd.to_datetime(t['doa']).dt.dayofyear.to_numpy()
            bands = t[CSV_BANDS].to_numpy(float) / 10000
            finite = np.isfinite(bands).all(1)
            X[i], DOY[i], M[i] = pack(bands, doy, (t['QA60'].to_numpy() == 0) & finite)
    else:
        sub = D if level == 'L2A' else D / f'{year}_{level}'
        index = pd.read_csv(sub / f'{region}.csv').set_index('id')['path']
        with h5py.File(sub / f'{region}.h5') as h:
            for i, pid in enumerate(ids):
                a = np.asarray(h[index[pid]])
                if level == 'L2A':
                    doy = pd.to_datetime(a[:, 0].astype('int64')).dayofyear.to_numpy()
                    bands = a[:, 1:11] / 10000
                    valid = (a[:, 11] < .5) & (a[:, 12] == 0) & (a[:, 13] == 0)
                else:
                    doy = pd.to_datetime(a[:, 16].astype('int64')).dayofyear.to_numpy()
                    bands = a[:, L1C_BANDS] / 10000
                    valid = a[:, 15] == 0
                X[i], DOY[i], M[i] = pack(bands, doy, valid)
    np.savez_compressed(out, x=X, doy=DOY, mask=M, ids=ids)
    print('SAVED', out.name, len(ids), 'mean obs', round(float(M.sum(1).mean()), 1), flush=True)


if __name__ == '__main__':
    import concurrent.futures
    jobs = [(2017, 'L2A', r) for r in ['frh01', 'frh02', 'frh03', 'frh04']] + \
           [(2017, 'L1C', 'frh01'), (2017, 'L1C', 'frh04'), (2018, 'L1C', 'frh01'), (2018, 'L1C', 'frh04')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for f in [ex.submit(extract, *j) for j in jobs]:
            f.result()
