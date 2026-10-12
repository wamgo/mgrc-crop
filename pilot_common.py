"""Portable I/O for this release; all output stays under the repository."""
from pathlib import Path
import os, sys, json, hashlib, time
ROOT=Path(__file__).resolve().parent
def setup(folder):
    root=ROOT
    for d in ['data','sources','results','figures','logs','cache']:(root/d).mkdir(parents=True,exist_ok=True)
    sys.dont_write_bytecode=True
    for k in ['TEMP','TMP','TMPDIR','TORCH_HOME','HF_HOME','XDG_CACHE_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','NUMBA_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR']:
        p=root/'cache'/k.lower();p.mkdir(exist_ok=True);os.environ[k]=str(p)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    return root
def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def download(url,path):
    import requests
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    meta=path.with_name(path.name+'.download.json')
    if path.exists() and meta.exists():return path
    temp=path.with_name(path.name+'.part')
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(20,90)) as r:
                r.raise_for_status();total=int(r.headers.get('Content-Length',0));n=0
                with temp.open('wb') as f:
                    for b in r.iter_content(1024*1024):
                        if b:f.write(b);n+=len(b)
                if total and not r.headers.get('Content-Encoding') and n!=total:raise IOError(f'Incomplete download {n}/{total}')
            temp.replace(path);save_json(meta,{'url':url,'bytes':n,'sha256':sha(path)})
            print('DOWNLOADED',path.name,n,flush=True);return path
        except Exception:
            if attempt==2:raise
            time.sleep(1+attempt)
def byte_range(url,start,end):
    import requests
    for attempt in range(4):
        try:
            # Distinct query prevents a proxy reusing a previous byte range.
            sep='&' if '?' in url else '?'
            r=requests.get(url+sep+f'pilot_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'},timeout=(20,90))
            if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
                raise IOError(f'Range not honored: {r.status_code} {r.headers.get("Content-Range")}')
            if len(r.content)!=end-start+1:raise IOError('Range length mismatch')
            return r.content
        except Exception:
            if attempt==3:raise
            time.sleep(attempt+1)
