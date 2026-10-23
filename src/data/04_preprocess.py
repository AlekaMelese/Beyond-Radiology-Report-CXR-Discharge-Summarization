from __future__ import annotations
import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import pandas as pd
try:
    from PIL import Image
except ImportError:
    sys.exit('ERROR: Pillow not installed. Run: pip install Pillow')
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('DATA_DIR', Path(__file__).resolve().parents[2] / 'data'))
CACHE_DIR = DATA_DIR / 'preprocessed' / 'images'
SUMMARY = DATA_DIR / 'preprocess_summary.csv'
SPLITS = ['train', 'val', 'test']
TARGET_SIZE = 512
JPEG_QUALITY = 95
N_WORKERS = 8
WS_RE = re.compile('\\s+')
DEID_RE = re.compile('_{2,}')

def clean_text(s: str | None) -> str:
    if not s:
        return ''
    s = DEID_RE.sub(' ', s)
    s = WS_RE.sub(' ', s)
    return s.strip()

def preprocess_one_image(src_path: str, dicom_id: str) -> tuple[str, bool, str]:
    out_path = CACHE_DIR / f'{dicom_id}.jpg'
    if out_path.exists():
        return (str(out_path), True, 'skip-cached')
    try:
        with Image.open(src_path) as im:
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
            im.save(out_path, 'JPEG', quality=JPEG_QUALITY, optimize=True)
        return (str(out_path), True, 'ok')
    except Exception as e:
        return (str(out_path), False, f'failed: {e}')

def dicom_id_from_path(path: str) -> str:
    return Path(path).stem

def process_split(split: str, pilot: int | None) -> dict:
    in_path = DATA_DIR / f'multimodal_{split}_clean.jsonl'
    out_path = DATA_DIR / f'multimodal_{split}_preprocessed.jsonl'
    if not in_path.exists():
        sys.exit(f'ERROR: {in_path} not found. Run earlier steps first.')
    print(f'[{split:>5s}] reading {in_path.name} ...', flush=True)
    records: list[dict] = []
    with in_path.open() as f:
        for i, line in enumerate(f):
            if pilot is not None and i >= pilot:
                break
            records.append(json.loads(line))
    n_total = len(records)
    print(f'        {n_total:,} records to process', flush=True)
    print(f'        resizing images (workers={N_WORKERS}) ...', flush=True)
    t0 = time.time()
    n_ok = n_fail = n_cached = 0
    new_paths: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        future_to_idx = {ex.submit(preprocess_one_image, r['cxr_image_path'], dicom_id_from_path(r['cxr_image_path'])): i for i, r in enumerate(records)}
        for j, fut in enumerate(as_completed(future_to_idx)):
            i = future_to_idx[fut]
            out_p, ok, note = fut.result()
            if ok:
                n_ok += 1
                if note == 'skip-cached':
                    n_cached += 1
                new_paths[i] = out_p
            else:
                n_fail += 1
                new_paths[i] = ''
            if (j + 1) % 2000 == 0:
                rate = (j + 1) / (time.time() - t0)
                print(f'        ... {j + 1:,}/{n_total:,} images at {rate:.0f} img/s', flush=True)
    elapsed = time.time() - t0
    print(f'        ✓ ok: {n_ok:,}  (cached: {n_cached:,})', flush=True)
    print(f'        ✗ failed: {n_fail:,}', flush=True)
    print(f'        elapsed: {elapsed:.1f}s ({n_total / elapsed:.0f} img/s)', flush=True)
    print(f'        writing {out_path.name} ...', flush=True)
    n_written = 0
    with out_path.open('w') as f:
        for i, r in enumerate(records):
            new_img = new_paths.get(i, '')
            if not new_img:
                continue
            r2 = dict(r)
            r2['cxr_image_path'] = new_img
            r2['input_note'] = clean_text(r.get('input_note'))
            r2['target_bhc'] = clean_text(r.get('target_bhc'))
            r2['radiology_report'] = clean_text(r.get('radiology_report'))
            f.write(json.dumps(r2, ensure_ascii=False) + '\n')
            n_written += 1
    print(f'        → {out_path}  ({n_written:,} records)', flush=True)
    return {'split': split, 'records_in': n_total, 'images_ok': n_ok, 'images_cached': n_cached, 'images_failed': n_fail, 'records_written': n_written, 'elapsed_seconds': round(elapsed, 1)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=int, default=None, help='If set, process only the first N records per split (smoke test).')
    args = ap.parse_args()
    if not DATA_DIR.exists():
        sys.exit(f'ERROR: {DATA_DIR} not found. Run 02_split_dataset.py first.')
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for split in SPLITS:
        summaries.append(process_split(split, args.pilot))
    pd.DataFrame(summaries).to_csv(SUMMARY, index=False)
    print()
    print(f'Wrote summary → {SUMMARY}')
    print(pd.DataFrame(summaries).to_string(index=False))
    print()
    print('✅ Step 4 done.  Splits now point at the cached 512x512 JPGs;')
    print('   text is cleaned.  Step 5 (PyTorch Dataset) will read these.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
