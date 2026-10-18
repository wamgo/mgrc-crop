"""EuroCropsML (Reuss & Macdonald 2025; Zenodo 10.5281/zenodo.15095445, CC BY-SA 4.0) features.

Second benchmark: cross-country transfer between Latvia (LV) and Estonia (EE), 2021, Sentinel-2 L1C
parcel medians as preprocessed (cloud-filtered) by the dataset authors.
Fine classes: the 24 HCAT3 codes with at least 200 parcels in both countries.
Parent groups: HCAT3 level 4, i.e. the first six digits of the code (13 groups).
Domains: source samples of each country (60,000 parcels), and every NUTS-3 region of the other country
with at least 25,000 parcels in the shared classes (capped at 60,000), so that each target keeps
20,000 test parcels and a calibration pool.
Outputs per domain (same layout as the BreizhCrops files): feat_2021_EC_<name>.npz with X (140),
cube (12x10), y, ids, xy (UTM 35N metres), and seq_2021_EC_<name>.npz for L-TAE.
"""
from pathlib import Path
import sys, io, json, zipfile, hashlib, collections, csv
from pilot_common import setup, save_json
ROOT = setup('direction_2_crop_hierarchy')
import numpy as np

SEED = 20261011
D = ROOT / 'data'
ZIP = D / 'eurocropsml' / 'preprocess.zip'
BANDS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]  # B2..B8A, B11, B12 in the S2_BANDS order of EuroCropsML
T_MAX = 96
MIN_SHARED, MIN_TARGET, CAP = 200, 25000, 60000


def members():
    z = zipfile.ZipFile(ZIP)
    out = []
    for n in z.namelist():
        if n.endswith('.npz'):
            reg, pid, h = n.split('/')[-1][:-4].split('_')
            out.append((n, reg, pid, h))
    return out


def taxonomy(mem):
    path = D / 'taxonomy_eurocrops.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf8'))
    names = {r['HCAT3_code']: r['HCAT3_name'] for r in
             csv.DictReader(open(ROOT / 'sources/eurocropsml/HCAT3.csv', encoding='utf8'))}
    cnt = collections.Counter((reg[:2], h) for _, reg, _, h in mem)
    fine = [h for h in {h for _, _, _, h in mem} if cnt[('LV', h)] >= MIN_SHARED and cnt[('EE', h)] >= MIN_SHARED]
    fine.sort(key=lambda h: (h[:6], -(cnt[('LV', h)] + cnt[('EE', h)])))
    groups = sorted({h[:6] for h in fine})
    tax = {'fine': fine, 'fine_names': [names[h] for h in fine],
           'parent_of_fine': [groups.index(h[:6]) for h in fine],
           'parents': [names[g + '0000'] for g in groups], 'hcat_level4_codes': groups,
           'counts': {h: [cnt[('LV', h)], cnt[('EE', h)]] for h in fine}}
    save_json(path, tax)
    return tax


def order(pid):
    return hashlib.sha256(f'{SEED}:EC:{pid}'.encode()).hexdigest()


def to_utm35(lon, lat):
    from pyproj import Transformer
    return Transformer.from_crs('EPSG:4326', 'EPSG:32635', always_xy=True).transform(lon, lat)


def extract(name, sel):
    fo, so = D / f'feat_2021_EC_{name}.npz', D / f'seq_2021_EC_{name}.npz'
    if fo.exists() and so.exists():
        return name
    tax = json.loads((D / 'taxonomy_eurocrops.json').read_text(encoding='utf8'))
    fine = {h: i for i, h in enumerate(tax['fine'])}
    z = zipfile.ZipFile(ZIP)
    n = len(sel)
    X = np.zeros((n, 140), np.float32); C = np.zeros((n, 12, 10), np.float32)
    S = np.zeros((n, T_MAX, 10), np.float16); DOY = np.zeros((n, T_MAX), np.int16); M = np.zeros((n, T_MAX), bool)
    Y = np.zeros(n, int); I = np.zeros(n, np.int64); LL = np.zeros((n, 2))
    keep = np.ones(n, bool)
    for i, (mname, reg, pid, h) in enumerate(sel):
        d = np.load(io.BytesIO(z.read(mname)))
        a = d['data'][:, BANDS].astype(float)
        ok = np.isfinite(a).all(1) & (a > -999).all(1) & (a.sum(1) > 0)
        if ok.sum() < 3:
            keep[i] = False
            continue
        a, dates = a[ok] / 10000, d['dates'][ok]
        a = np.clip(a, 0, 2)
        months = dates.astype('datetime64[M]').astype(int) % 12 + 1
        doy = (dates - dates.astype('datetime64[Y]')).astype(int) + 1
        med = np.median(a, axis=0)
        monthly = np.stack([np.median(a[months == m], axis=0) if np.any(months == m) else med for m in range(1, 13)])
        X[i] = np.r_[monthly.ravel(), np.quantile(a, [.1, .9], axis=0).ravel()]
        C[i] = monthly
        idx = np.arange(len(a))
        if len(idx) > T_MAX:
            idx = idx[np.round(np.linspace(0, len(idx) - 1, T_MAX)).astype(int)]
        S[i, :len(idx)] = a[idx]; DOY[i, :len(idx)] = doy[idx]; M[i, :len(idx)] = True
        Y[i] = fine[h]; I[i] = int(pid); LL[i] = d['center']
    X, C, S, DOY, M, Y, I, LL = (v[keep] for v in (X, C, S, DOY, M, Y, I, LL))
    ex, ey = to_utm35(LL[:, 0], LL[:, 1])
    xy = np.stack([ex, ey], 1)
    np.savez_compressed(fo, X=X, cube=C, y=Y, ids=I, xy=xy)
    np.savez_compressed(so, x=S, doy=DOY, mask=M, ids=I)
    print('SAVED', name, len(Y), 'mean obs', round(float(M.sum(1).mean()), 1), flush=True)
    return name


if __name__ == '__main__':
    import concurrent.futures
    mem = members()
    tax = taxonomy(mem)
    fine = set(tax['fine'])
    mem = [m for m in mem if m[3] in fine and m[1][:2] in ('LV', 'EE')]
    by = collections.defaultdict(list)
    for m in mem:
        by[m[1][:2]].append(m)
        by[m[1]].append(m)
    domains = {c: sorted(by[c], key=lambda m: order(m[2]))[:CAP] for c in ('LV', 'EE')}
    for r in sorted(k for k in by if len(k) > 2):
        if len(by[r]) >= MIN_TARGET:
            domains[r] = sorted(by[r], key=lambda m: order(m[2]))[:CAP]
    save_json(D / 'eurocrops_domains.json', {k: len(v) for k, v in domains.items()})
    print({k: len(v) for k, v in domains.items()}, flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for f in concurrent.futures.as_completed([ex.submit(extract, k, v) for k, v in domains.items()]):
            f.result()
