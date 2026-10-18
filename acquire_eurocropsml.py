"""Download EuroCropsML (Reuss & Macdonald, Zenodo 10.5281/zenodo.15095445, CC BY-SA 4.0)."""
from pathlib import Path
import sys
from pilot_common import setup, download
ROOT = setup('direction_2_crop_hierarchy')
D = ROOT / 'data' / 'eurocropsml'
D.mkdir(parents=True, exist_ok=True)
BASE = 'https://zenodo.org/api/records/15095445/files/'
for name in ['split.zip', 'preprocess.zip']:
    download(BASE + name + '/content', D / name)
    print('READY', name, flush=True)
