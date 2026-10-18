"""Fetch remaining BreizhCrops regions (2017 L2A), L1C 2017/2018 for cross-year, and parcel shapefiles."""
from pathlib import Path
import sys,tarfile,shutil,concurrent.futures
from pilot_common import setup,download,save_json,sha
ROOT=setup('direction_2_crop_hierarchy')
BASE='https://breizhcrops.s3.eu-central-1.amazonaws.com/'
D=ROOT/'data'
def h5(year,level,region):
    sub=D/f'{year}_{level}';sub.mkdir(exist_ok=True)
    if year==2017 and level=='L2A':sub=D
    download(BASE+f'{year}/{level}/{region}.csv',sub/f'{region}.csv')
    archive=download(BASE+f'{year}/{level}/{region}.h5.tar.gz',sub/f'{region}.h5.tar.gz')
    dest=sub/f'{region}.h5'
    if not dest.exists():
        with tarfile.open(archive) as tf:
            m=[m for m in tf.getmembers() if m.isfile() and m.name.endswith('.h5')];assert len(m)==1
            with tf.extractfile(m[0]) as s,dest.open('wb') as o:shutil.copyfileobj(s,o)
        save_json(dest.with_suffix('.sha256.json'),{'sha256':sha(dest)})
    return f'{year}/{level}/{region}'
def raw2018(region):
    sub=D/'2018_L1C';sub.mkdir(exist_ok=True)
    download(BASE+f'2018/L1C/{region}.csv',sub/f'{region}.csv')
    download(BASE+f'2018/L1C/{region}.zip',sub/f'{region}.zip')
    return f'2018/L1C/{region}'
def shp(year,region):
    sub=D/f'shp_{year}';sub.mkdir(exist_ok=True)
    a=download(BASE+f'{year}/shp/{region}.tar.gz',sub/f'{region}.tar.gz')
    out=sub/region
    if not out.exists():
        out.mkdir()
        with tarfile.open(a) as tf:tf.extractall(out,filter='data')
    return f'shp/{year}/{region}'
if __name__=='__main__':
    jobs=[(shp,2017,r) for r in ['frh01','frh02','frh03','frh04']]+[(shp,2018,r) for r in ['frh01','frh04']]
    jobs+=[(h5,2017,'L2A','frh02'),(h5,2017,'L2A','frh03'),(h5,2017,'L1C','frh01'),(h5,2017,'L1C','frh04'),(raw2018,'frh01'),(raw2018,'frh04')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
        for r in p.map(lambda j:j[0](*j[1:]),jobs):print('READY',r,flush=True)
