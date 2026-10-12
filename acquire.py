from pathlib import Path
import sys,tarfile,concurrent.futures
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pilot_common import setup,download,save_json,sha
ROOT=setup('direction_2_crop_hierarchy')
BASE='https://breizhcrops.s3.eu-central-1.amazonaws.com/'
def get(region):
    download(BASE+f'2017/L2A/{region}.csv',ROOT/'data'/f'{region}.csv')
    archive=download(BASE+f'2017/L2A/{region}.h5.tar.gz',ROOT/'data'/f'{region}.h5.tar.gz')
    dest=ROOT/'data'/f'{region}.h5'
    if not dest.exists():
        with tarfile.open(archive) as tf:
            members=[m for m in tf.getmembers() if m.isfile() and m.name.endswith('.h5')]
            assert len(members)==1
            with tf.extractfile(members[0]) as src,dest.open('wb') as out:
                import shutil;shutil.copyfileobj(src,out)
        save_json(dest.with_suffix('.sha256.json'),{'sha256':sha(dest)})
    return region
if __name__=='__main__':
    download(BASE+'classmapping.csv',ROOT/'data/classmapping.csv')
    download('https://raw.githubusercontent.com/dl4sits/BreizhCrops/master/breizhcrops/datasets/breizhcrops.py',ROOT/'sources/official_loader.py')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:
        for r in p.map(get,['frh01','frh04']):print('READY',r,flush=True)
