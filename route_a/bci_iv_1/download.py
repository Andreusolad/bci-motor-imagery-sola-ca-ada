"""Download BCI Competition IV Data Set 1 and the true labels of its evaluation sets.

Downloads three archives into $BCI_DATA/bci_iv_1/downloads/ (BCI_DATA defaults to
<repo>/data) and extracts their files, without subfolders, into $BCI_DATA/bci_iv_1/:
  1. BCICIV_1_mat.zip       the 14 recordings (7 calibration + 7 evaluation), 100 Hz.
  2. true_labels.zip        true labels of the evaluation recordings (MAT).
  3. true_labels_txt.zip    the same labels as TXT, used as an independent parsing check.

The label archives are not linked from the dataset download page; they are listed on
the competition results page under "True Labels of Competition's Evaluation Sets".

Archives already on disk are not downloaded again. The SHA-256 and size of every
archive and the list of extracted files are written to outputs/download.json. If a
download fails the script exits with an error.

Dataset citation requested by the organizers:
  Benjamin Blankertz, Guido Dornhege, Matthias Krauledat, Klaus-Robert Muller, Gabriel
  Curio. The non-invasive Berlin Brain-Computer Interface: Fast acquisition of
  effective performance in untrained subjects. NeuroImage 37(2):539-550, 2007.

Usage:
    python download.py
    python download.py --solo-verificar     # download nothing: re-hash and re-extract
                                            # the archives on disk, fail if one is missing
"""
from __future__ import annotations
import argparse
import os
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('BCI_DATA', HERE.parents[1] / 'data')) / 'bci_iv_1'
DL = RAW / 'downloads'
OUT = HERE / 'outputs'

BASE_DATOS = 'https://www.bbci.de/competition/download/competition_iv/'
BASE_RES = 'https://www.bbci.de/competition/iv/results/'

FUENTES = [
    dict(nombre='BCICIV_1_mat.zip', url=BASE_DATOS + 'BCICIV_1_mat.zip',
         procedencia='Competition IV download page (100 Hz, Matlab)',
         extraer=True),
    dict(nombre='true_labels.zip', url=BASE_RES + 'ds1/true_labels.zip',
         procedencia='Competition IV results page, "True Labels ... Data sets 1: [MAT]"',
         extraer=True),
    dict(nombre='true_labels_txt.zip', url=BASE_RES + 'ds1/true_labels_txt.zip',
         procedencia='Competition IV results page, "True Labels ... Data sets 1: [TXT]"',
         extraer=True),
]

CITA = ('Benjamin Blankertz, Guido Dornhege, Matthias Krauledat, Klaus-Robert Muller, '
        'Gabriel Curio. The non-invasive Berlin Brain-Computer Interface: Fast '
        'acquisition of effective performance in untrained subjects. NeuroImage '
        '37(2):539-550, 2007.')


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in chunks of `chunk` bytes."""
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def descargar(url: str, destino: Path) -> dict:
    """Download `url` to `destino`, printing progress.

    Data are written to a '.parcial' file that is renamed only when the download is
    complete, so a partially written file is never taken for a finished one.
    """
    tmp = destino.with_suffix(destino.suffix + '.parcial')
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as r:
        total = int(r.headers.get('Content-Length', 0))
        leido = 0
        with open(tmp, 'wb') as fh:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                fh.write(b)
                leido += len(b)
                if total and (leido // (1 << 20)) % 20 == 0:
                    print(f'    {leido/1e6:8.1f} / {total/1e6:.1f} MB '
                          f'({100*leido/total:5.1f} %)', flush=True)
    tmp.replace(destino)
    seg = time.time() - t0
    print(f'    -> {destino.name}  {destino.stat().st_size/1e6:.1f} MB in {seg:.0f}s',
          flush=True)
    return dict(segundos=round(seg, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solo-verificar', action='store_true')
    a = ap.parse_args()

    t0 = time.time()
    print('=' * 80)
    print('Download of BCI Competition IV, Data Set 1')
    print('=' * 80, flush=True)
    for d in (RAW, DL, OUT):
        d.mkdir(parents=True, exist_ok=True)

    res = dict(cita_obligatoria=CITA, base_datos=BASE_DATOS, base_resultados=BASE_RES,
               ficheros=[], extraidos=[])
    for f in FUENTES:
        dest = DL / f['nombre']
        print(f'\n[{f["nombre"]}]  {f["url"]}', flush=True)
        if dest.exists():
            print(f'    already on disk ({dest.stat().st_size/1e6:.1f} MB), not downloaded',
                  flush=True)
            tiempos = dict(segundos=0.0, reutilizado=True)
        elif a.solo_verificar:
            raise SystemExit(f'{dest} is missing and --solo-verificar was given')
        else:
            try:
                tiempos = descargar(f['url'], dest)
            except Exception as e:                                    # noqa: BLE001
                raise SystemExit(
                    f'\n[STOP] could not download {f["url"]}: {e}\n'
                    f'  Check the network connection and the URL.\n'
                    f'  No substitute data are used.')
        h = sha256(dest)
        print(f'    sha256 {h}', flush=True)
        res['ficheros'].append(dict(nombre=f['nombre'], url=f['url'],
                                    procedencia=f['procedencia'],
                                    bytes=int(dest.stat().st_size), sha256=h,
                                    **tiempos))

    # ---------- extraction ----------
    print('\n--- extracting to $BCI_DATA/bci_iv_1/ ---', flush=True)
    for f in FUENTES:
        if not f['extraer']:
            continue
        with zipfile.ZipFile(DL / f['nombre']) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                # Flatten the subfolders of the label archives. Both label archives
                # contain a read_me.txt; prefix it with the archive name so that one
                # does not overwrite the other (the read_me states that the evaluation
                # of subject 'a' is truncated).
                nombre = Path(info.filename).name
                if nombre.lower().startswith('read_me'):
                    nombre = f'{Path(f["nombre"]).stem}__{nombre}'
                destino = RAW / nombre
                with z.open(info) as src, open(destino, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                res['extraidos'].append(dict(zip=f['nombre'], interno=info.filename,
                                             destino=nombre,
                                             bytes=int(destino.stat().st_size)))
    for e in sorted(res['extraidos'], key=lambda d: d['destino']):
        print(f'  {e["destino"]:34s} {e["bytes"]/1e6:8.2f} MB   (from {e["zip"]})',
              flush=True)
    res['n_extraidos'] = len(res['extraidos'])
    res['directorio_raw'] = str(RAW)
    res['bytes_totales_descargados'] = sum(x['bytes'] for x in res['ficheros'])

    (OUT / 'download.json').write_text(
        json.dumps(res, indent=1), encoding='utf-8')
    print(f'\n[OK] {time.time()-t0:.0f}s  {res["n_extraidos"]} files extracted, '
          f'{res["bytes_totales_descargados"]/1e6:.1f} MB downloaded '
          f'-> outputs/download.json')


if __name__ == '__main__':
    main()
