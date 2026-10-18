"""Fetch the AEZ 22190 (Baltic) GeoTIFFs of ESA WorldCereal 2021 v100 (Van Tricht et al., 2023;
Zenodo 10.5281/zenodo.7875105, CC BY 4.0) without downloading the 7-18 GB archives.

Each archive is a zip with one GeoTIFF per agro-ecological zone. We read the zip central directory and
the single member we need through HTTP range requests, then inflate it locally.
"""
from pathlib import Path
import struct, sys, zlib
from pilot_common import setup, save_json, byte_range
ROOT = setup('direction_2_crop_hierarchy')
import requests

AEZ = '22190'
BASE = 'https://zenodo.org/api/records/7875105/files/'
ARCHIVES = ['WorldCereal_2021_tc-annual_temporarycrops_classification.zip',
            'WorldCereal_2021_tc-wintercereals_wintercereals_classification.zip',
            'WorldCereal_2021_tc-springcereals_springcereals_classification.zip',
            'WorldCereal_2021_tc-maize-main_maize_classification.zip']
OUT = ROOT / 'data' / 'worldcereal'
OUT.mkdir(parents=True, exist_ok=True)


def resolve(url):
    r = requests.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.url, int(r.headers['Content-Length'])


def central_directory(url, size):
    tail = byte_range(url, size - 65557, size - 1) if size > 65557 else byte_range(url, 0, size - 1)
    i = tail.rfind(b'PK\x05\x06')
    if i < 0:
        raise IOError('EOCD not found')
    cd_size, cd_off = struct.unpack('<II', tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:  # zip64
        j = tail.rfind(b'PK\x06\x07')
        z64_off = struct.unpack('<Q', tail[j + 8:j + 16])[0]
        z64 = byte_range(url, z64_off, z64_off + 55)
        cd_size, cd_off = struct.unpack('<QQ', z64[40:56])
    return byte_range(url, cd_off, cd_off + cd_size - 1)


def entries(cd):
    p = 0
    while p < len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[p + 10:p + 12])
        csize, usize = struct.unpack('<II', cd[p + 20:p + 28])
        nlen, elen, clen = struct.unpack('<HHH', cd[p + 28:p + 34])
        off, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode('utf8')
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):  # zip64 extended information
            hid, hlen = struct.unpack('<HH', extra[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack('<' + 'Q' * (hlen // 8), extra[q + 4:q + 4 + hlen]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if off == 0xFFFFFFFF:
                    off = vals.pop(0)
            q += 4 + hlen
        yield name, method, csize, usize, off
        p += 46 + nlen + elen + clen


def fetch_member(url, method, csize, off, dest):
    head = byte_range(url, off, off + 29)
    nlen, elen = struct.unpack('<HH', head[26:30])
    start = off + 30 + nlen + elen
    chunk = 64 * 2 ** 20
    dec = zlib.decompressobj(-15) if method == 8 else None
    tmp = dest.with_suffix('.part')
    with tmp.open('wb') as f:
        for a in range(start, start + csize, chunk):
            b = min(a + chunk, start + csize) - 1
            data = byte_range(url, a, b)
            f.write(dec.decompress(data) if dec else data)
            print(f'  {dest.name}: {(b - start + 1) / csize:6.1%}', flush=True)
        if dec:
            f.write(dec.flush())
    tmp.replace(dest)


if __name__ == '__main__':
    log = {}
    for arc in ARCHIVES:
        url, size = resolve(BASE + arc + '/content')
        members = [e for e in entries(central_directory(url, size)) if Path(e[0]).name.startswith(AEZ + '_')]
        assert len(members) == 1, (arc, [m[0] for m in members])
        name, method, csize, usize, off = members[0]
        dest = OUT / Path(name).name
        if not dest.exists():
            print('FETCH', name, round(csize / 2 ** 20), 'MB compressed', flush=True)
            fetch_member(url, method, csize, off, dest)
        assert dest.stat().st_size == usize, (dest, dest.stat().st_size, usize)
        log[arc] = {'member': name, 'compressed': csize, 'size': usize}
        print('READY', dest.name, flush=True)
    save_json(OUT / 'acquisition.json', log)
