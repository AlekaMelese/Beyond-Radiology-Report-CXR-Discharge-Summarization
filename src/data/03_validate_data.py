from __future__ import annotations
import json
import sys
import os
from pathlib import Path
from collections import Counter
import pandas as pd
try:
    from PIL import Image
except ImportError:
    sys.exit('ERROR: Pillow not installed. Run: pip install Pillow')
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('DATA_DIR', Path(__file__).resolve().parents[2] / 'data'))
SPLITS = ['train', 'val', 'test']
REPORT = DATA_DIR / 'validation_report.csv'
BROKEN = DATA_DIR / 'broken_records.jsonl'
MIN_IMG_SIDE = 100
MIN_TEXT_LEN = 10

def validate_split(split: str) -> tuple[dict, list[dict]]:
    in_path = DATA_DIR / f'multimodal_{split}.jsonl'
    print(f'[{split:>5s}] reading {in_path.name} ...', flush=True)
    n_total = 0
    n_ok = 0
    n_img_missing = 0
    n_img_broken = 0
    n_img_too_small = 0
    n_text_missing = 0
    img_widths = []
    img_heights = []
    mode_counter = Counter()
    note_chars = []
    target_chars = []
    report_chars = []
    broken: list[dict] = []
    for line in in_path.open():
        n_total += 1
        rec = json.loads(line)
        path = rec.get('cxr_image_path', '')
        note = rec.get('input_note') or ''
        tgt = rec.get('target_bhc') or ''
        rpt = rec.get('radiology_report') or ''
        if len(note) < MIN_TEXT_LEN or len(tgt) < MIN_TEXT_LEN:
            n_text_missing += 1
            broken.append({'reason': 'text_too_short', 'subject_id': rec.get('subject_id'), 'note_len': len(note), 'tgt_len': len(tgt)})
            continue
        p = Path(path)
        if not p.exists():
            n_img_missing += 1
            broken.append({'reason': 'image_missing', 'subject_id': rec.get('subject_id'), 'path': path})
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
        except Exception as e:
            n_img_broken += 1
            broken.append({'reason': 'image_open_failed', 'subject_id': rec.get('subject_id'), 'path': path, 'error': str(e)})
            continue
        if w < MIN_IMG_SIDE or h < MIN_IMG_SIDE:
            n_img_too_small += 1
            broken.append({'reason': 'image_too_small', 'subject_id': rec.get('subject_id'), 'path': path, 'w': w, 'h': h})
            continue
        n_ok += 1
        img_widths.append(w)
        img_heights.append(h)
        mode_counter[mode] += 1
        note_chars.append(len(note))
        target_chars.append(len(tgt))
        report_chars.append(len(rpt))
        if n_total % 5000 == 0:
            print(f'        ... {n_total:,} checked, {n_ok:,} ok', flush=True)

    def pct(n):
        return f'{n / n_total * 100:5.1f}%'
    print(f'        total: {n_total:,}')
    print(f'        ✓ ok:                {n_ok:>6,d} ({pct(n_ok)})')
    print(f'        ✗ image_missing:     {n_img_missing:>6,d} ({pct(n_img_missing)})')
    print(f'        ✗ image_broken:      {n_img_broken:>6,d} ({pct(n_img_broken)})')
    print(f'        ✗ image_too_small:   {n_img_too_small:>6,d} ({pct(n_img_too_small)})')
    print(f'        ✗ text_too_short:    {n_text_missing:>6,d} ({pct(n_text_missing)})')

    def stats(xs):
        if not xs:
            return (0, 0, 0, 0)
        s = pd.Series(xs)
        return (int(s.min()), int(s.median()), int(s.mean()), int(s.max()))
    w_min, w_med, w_mean, w_max = stats(img_widths)
    h_min, h_med, h_mean, h_max = stats(img_heights)
    n_min, n_med, n_mean, n_max = stats(note_chars)
    t_min, t_med, t_mean, t_max = stats(target_chars)
    r_min, r_med, r_mean, r_max = stats(report_chars)
    summary = {'split': split, 'n_total': n_total, 'n_ok': n_ok, 'n_image_missing': n_img_missing, 'n_image_broken': n_img_broken, 'n_image_too_small': n_img_too_small, 'n_text_too_short': n_text_missing, 'img_width_min': w_min, 'img_width_median': w_med, 'img_width_mean': w_mean, 'img_width_max': w_max, 'img_height_median': h_med, 'img_height_mean': h_mean, 'img_height_max': h_max, 'img_modes': ','.join((f'{k}={v}' for k, v in mode_counter.most_common())), 'input_note_chars_median': n_med, 'input_note_chars_mean': n_mean, 'input_note_chars_max': n_max, 'target_bhc_chars_median': t_med, 'target_bhc_chars_mean': t_mean, 'target_bhc_chars_max': t_max, 'report_chars_median': r_med, 'report_chars_mean': r_mean, 'report_chars_max': r_max}
    return (summary, broken)

def main() -> int:
    if not DATA_DIR.exists():
        sys.exit(f'ERROR: {DATA_DIR} not found. Run 02_split_dataset.py first.')
    all_summaries = []
    all_broken = []
    for split in SPLITS:
        s, b = validate_split(split)
        all_summaries.append(s)
        all_broken.extend(b)
    pd.DataFrame(all_summaries).to_csv(REPORT, index=False)
    with BROKEN.open('w') as f:
        for b in all_broken:
            f.write(json.dumps(b) + '\n')
    print()
    print(f'Wrote per-split report → {REPORT}')
    print(f'Wrote broken-records list → {BROKEN}  ({len(all_broken):,} entries)')
    if all_broken:
        print('⚠️  Some records failed validation — see broken_records.jsonl for details.')
        from collections import Counter
        reasons = Counter((b['reason'] for b in all_broken))
        print('    Top failure reasons:')
        for r, n in reasons.most_common():
            print(f'      {r}: {n:,}')
    else:
        print('✅ All records pass validation — data is ready for training.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
