"""Feature extraction for the hierarchical-risk experiments.

Fine labels: raw RPG crop codes (CODE_CULTU) with enough parcels in every region.
Parent labels: official RPG group (CODE_GROUP), read from the 2018 parcel register; it is the
French administration's grouping, not one defined for this study.
Per parcel: monthly median of 10 Sentinel-2 bands over clear observations (12x10) plus 10/90%
quantiles (2x10) -> 140 features; also the 12x10 monthly cube for the temporal CNN; parcel centroid.
"""
from pathlib import Path
import sys, hashlib, json, collections
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'deps'))
from pilot_common import setup, save_json
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np, pandas as pd, h5py, shapefile

SEED = 20261011
D = ROOT / 'data'
FINE = ['BTH', 'MIE', 'MIS', 'ORH', 'ORP', 'TTH', 'AVH', 'MCR', 'SRS', 'CZH', 'J6S', 'J5M',
        'FET', 'BVF', 'PPH', 'PRL', 'PTR', 'RGA', 'PTC', 'FLA', 'SNE', 'BTA', 'VRG']
GROUP_NAME = {1: 'soft wheat', 2: 'maize', 3: 'barley', 4: 'other cereals', 5: 'rapeseed', 11: 'fallow',
              16: 'forage', 18: 'permanent grassland', 19: 'temporary grassland', 25: 'vegetables/flowers',
              28: 'miscellaneous', 20: 'orchards'}


def taxonomy():
    path = D / 'taxonomy.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    s = shapefile.Reader(str(D / 'shp_2018/frh01/frh01'))
    m = collections.defaultdict(collections.Counter)
    for code, grp in s.iterRecords(fields=['CODE_CULTU', 'CODE_GROUP']):
        m[code][int(grp)] += 1
    code_group = {c: m[c].most_common(1)[0][0] for c in FINE}
    groups = sorted(set(code_group.values()))
    tax = {'fine': FINE, 'parent_of_fine': [groups.index(code_group[c]) for c in FINE],
           'parents': [GROUP_NAME[g] for g in groups], 'rpg_group_codes': groups,
           'purity': {c: m[c][code_group[c]] / sum(m[c].values()) for c in FINE}}
    save_json(path, tax)
    return tax


def centroids(year, region):
    s = shapefile.Reader(str(D / f'shp_{year}/{region}/{region}'))
    key = 'ID' if year == 2017 else 'ID_PARCEL'
    ids = [int(r[0]) for r in s.iterRecords(fields=[key])]
    xy = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (sh.bbox for sh in s.iterShapes())])
    return dict(zip(ids, map(tuple, xy)))


def extract(year, level, region, cap=60000):
    out = D / f'feat_{year}_{level}_{region}.npz'
    if out.exists():
        return dict(np.load(out))
    sub = D if (year, level) == (2017, 'L2A') else D / f'{year}_{level}'
    tax = taxonomy()
    df = pd.read_csv(sub / f'{region}.csv')
    df = df[df.CODE_CULTU.isin(tax['fine'])].copy()
    df['order'] = df.id.map(lambda x: hashlib.sha256(f'{SEED}:{year}:{x}'.encode()).hexdigest())
    df = df.sort_values('order').head(cap)
    cxy = centroids(year, region)
    if level == 'L2A':
        band_idx, qa = slice(1, 11), lambda a: (a[:, 11] < .5) & (a[:, 12] == 0) & (a[:, 13] == 0)
        date_col = 0
    else:  # L1C h5 columns are alphabetical: B1,B10,B11,B12,B2,B3,B4,B5,B6,B7,B8,B8A,B9,QA10,QA20,QA60,doa
        band_idx = [4, 5, 6, 7, 8, 9, 10, 11, 2, 3]  # B2..B8A, B11, B12 (same order as L2A)
        qa = lambda a: a[:, 15] == 0  # QA60: no opaque/cirrus flag
        date_col = 16
    X, C, Y, I, XY = [], [], [], [], []
    with h5py.File(sub / f'{region}.h5') as h:
        for i, r in enumerate(df.itertuples()):
            a = np.asarray(h[r.path])
            if len(a) < 3:
                continue
            dates = pd.to_datetime(a[:, date_col].astype('int64'))
            months = dates.month.to_numpy()
            bands = np.clip(a[:, band_idx] / 10000, 0, 2).astype(np.float32)
            valid = qa(a)
            if valid.sum() < 3:
                valid = np.ones(len(a), bool)
            med = np.median(bands[valid], axis=0)
            monthly = np.stack([np.median(bands[valid & (months == m)], axis=0) if np.any(valid & (months == m)) else med
                                for m in range(1, 13)])
            X.append(np.r_[monthly.ravel(), np.quantile(bands[valid], [.1, .9], axis=0).ravel()])
            C.append(monthly)
            Y.append(tax['fine'].index(r.CODE_CULTU))
            I.append(int(r.id))
            XY.append(cxy.get(int(r.id), (np.nan, np.nan)))
            if (i + 1) % 10000 == 0:
                print('FEATURES', year, level, region, i + 1, flush=True)
    res = {'X': np.array(X, np.float32), 'cube': np.array(C, np.float32), 'y': np.array(Y, int),
           'ids': np.array(I, np.int64), 'xy': np.array(XY, float)}
    np.savez_compressed(out, **res)
    print('SAVED', out.name, len(Y), 'missing xy', int(np.isnan(res['xy'][:, 0]).sum()), flush=True)
    return res


def extract_2018_zip(region, cap=60000):
    """2018 L1C is only distributed as per-parcel CSVs in a zip; read the sampled parcels directly."""
    import zipfile, io
    out = D / f'feat_2018_L1C_{region}.npz'
    if out.exists():
        return dict(np.load(out))
    tax = taxonomy()
    df = pd.read_csv(D / '2018_L1C' / f'{region}.csv')
    df = df[df.CODE_CULTU.isin(tax['fine'])].copy()
    df['order'] = df.id.map(lambda x: hashlib.sha256(f'{SEED}:2018:{x}'.encode()).hexdigest())
    df = df.sort_values('order')
    zf = zipfile.ZipFile(D / '2018_L1C' / f'{region}.zip')
    names = {Path(n).stem: n for n in zf.namelist() if n.endswith('.csv')}
    X, C, Y, I = [], [], [], []
    cxy = centroids(2018, region)
    bands_needed = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    for r in df.itertuples():
        n = names.get(str(r.id))
        if n is None:
            continue
        t = pd.read_csv(io.BytesIO(zf.read(n)))
        if len(t) < 4:
            continue
        dates = pd.to_datetime(t['doa'])
        months = dates.dt.month.to_numpy()
        bands = np.clip(t[bands_needed].to_numpy(float) / 10000, 0, 2).astype(np.float32)
        finite = np.isfinite(bands).all(1)
        valid = (t['QA60'].to_numpy() == 0) & finite
        if valid.sum() < 3:
            valid = finite
        if valid.sum() < 3:
            continue
        med = np.median(bands[valid], axis=0)
        monthly = np.stack([np.median(bands[valid & (months == m)], axis=0) if np.any(valid & (months == m)) else med
                            for m in range(1, 13)])
        X.append(np.r_[monthly.ravel(), np.quantile(bands[valid], [.1, .9], axis=0).ravel()])
        C.append(monthly)
        Y.append(tax['fine'].index(r.CODE_CULTU))
        I.append(int(r.id))
        if len(Y) % 10000 == 0:
            print('FEATURES 2018', region, len(Y), flush=True)
        if len(Y) >= cap:
            break
    res = {'X': np.array(X, np.float32), 'cube': np.array(C, np.float32), 'y': np.array(Y, int),
           'ids': np.array(I, np.int64), 'xy': np.array([cxy.get(i, (np.nan, np.nan)) for i in I], float)}
    np.savez_compressed(out, **res)
    print('SAVED', out.name, len(Y), flush=True)
    return res


if __name__ == '__main__':
    import concurrent.futures
    if sys.argv[1:2] == ['2018']:
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as p:
            list(p.map(extract_2018_zip, ['frh01', 'frh04']))
        sys.exit()
    jobs = [tuple(a.split(':')) for a in sys.argv[1:]] or [('2017', 'L2A', r) for r in ['frh01', 'frh02', 'frh03', 'frh04']]
    taxonomy()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
        futs = [p.submit(extract, int(y), l, r) for y, l, r in jobs]
        for f in futs:
            f.result()
