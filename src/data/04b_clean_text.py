from __future__ import annotations
import argparse
import json
import re
import sys
from datetime import datetime
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('DATA_DIR', Path(__file__).resolve().parents[2] / 'data'))
DEFAULT_MAX_NOTE_CHARS = 5000
DEFAULT_MIN_TARGET_CHARS = 200
DEFAULT_MAX_TARGET_CHARS = 4000
PHI_PLACEHOLDER = re.compile('_{3,}|\\[\\*\\*[^]]*\\*\\*\\]')
WHITESPACE_RUN = re.compile('\\s+')
SECTION_MARKER = re.compile('^\\s*#+\\s*[A-Z][^\\n:]{0,80}:', re.MULTILINE)

def clean_note(text: str, max_chars: int) -> str:
    out = PHI_PLACEHOLDER.sub(' ', text)
    out = WHITESPACE_RUN.sub(' ', out).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out

def clean_target(text: str) -> str:
    out = PHI_PLACEHOLDER.sub(' ', text)
    out = SECTION_MARKER.sub('', out)
    out = WHITESPACE_RUN.sub(' ', out).strip()
    return out

def process_split(split: str, in_path: Path, out_path: Path, max_note_chars: int, min_target_chars: int, max_target_chars: int) -> dict:
    n_in = n_out = 0
    drop_short_target = drop_long_target = drop_missing_image = 0
    sum_note_in = sum_note_out = sum_tgt_in = sum_tgt_out = 0
    with in_path.open() as fin, out_path.open('w') as fout:
        for line in fin:
            r = json.loads(line)
            n_in += 1
            note_in = r.get('input_note', '')
            target_in = r.get('target_bhc', '')
            sum_note_in += len(note_in)
            sum_tgt_in += len(target_in)
            img_path = Path(r.get('cxr_image_path', ''))
            if not img_path.exists():
                drop_missing_image += 1
                continue
            note = clean_note(note_in, max_note_chars)
            target = clean_target(target_in)
            if len(target) < min_target_chars:
                drop_short_target += 1
                continue
            if len(target) > max_target_chars:
                drop_long_target += 1
                continue
            r['input_note'] = note
            r['target_bhc'] = target
            r['_clean'] = {'orig_note_chars': len(note_in), 'orig_target_chars': len(target_in)}
            fout.write(json.dumps(r) + '\n')
            n_out += 1
            sum_note_out += len(note)
            sum_tgt_out += len(target)
    return {'split': split, 'in_path': str(in_path), 'out_path': str(out_path), 'n_in': n_in, 'n_out': n_out, 'drop_short_target': drop_short_target, 'drop_long_target': drop_long_target, 'drop_missing_image': drop_missing_image, 'mean_note_chars_before': round(sum_note_in / max(n_in, 1), 1), 'mean_note_chars_after': round(sum_note_out / max(n_out, 1), 1), 'mean_target_chars_before': round(sum_tgt_in / max(n_in, 1), 1), 'mean_target_chars_after': round(sum_tgt_out / max(n_out, 1), 1)}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', nargs='+', default=['train', 'val', 'test'], choices=['train', 'val', 'test'])
    ap.add_argument('--max-note-chars', type=int, default=DEFAULT_MAX_NOTE_CHARS)
    ap.add_argument('--min-target-chars', type=int, default=DEFAULT_MIN_TARGET_CHARS)
    ap.add_argument('--max-target-chars', type=int, default=DEFAULT_MAX_TARGET_CHARS)
    args = ap.parse_args()
    summary = {'timestamp': datetime.now().strftime('%Y%m%d-%H%M%S'), 'max_note_chars': args.max_note_chars, 'min_target_chars': args.min_target_chars, 'max_target_chars': args.max_target_chars, 'splits': {}}
    for split in args.split:
        in_path = DATA_DIR / f'multimodal_{split}_preprocessed.jsonl'
        out_path = DATA_DIR / f'multimodal_{split}_cleaned.jsonl'
        if not in_path.exists():
            sys.exit(f'Input missing: {in_path}.  Run 04_preprocess.py first.')
        print(f'[{split}] {in_path.name} → {out_path.name}')
        s = process_split(split, in_path, out_path, args.max_note_chars, args.min_target_chars, args.max_target_chars)
        summary['splits'][split] = s
        print(f"   in={s['n_in']:,}  out={s['n_out']:,}  drop(short/long/img)={s['drop_short_target']}/{s['drop_long_target']}/{s['drop_missing_image']}  note_chars {s['mean_note_chars_before']}→{s['mean_note_chars_after']}  tgt_chars {s['mean_target_chars_before']}→{s['mean_target_chars_after']}")
    (DATA_DIR / 'clean_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\n✅ Cleaned files at {DATA_DIR}/multimodal_*_cleaned.jsonl')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
