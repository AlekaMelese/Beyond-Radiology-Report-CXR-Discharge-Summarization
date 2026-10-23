from __future__ import annotations
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / 'Data' / 'multimodal_records.jsonl'
OUT_DIR = HERE / 'data'
TRAIN_FILE = OUT_DIR / 'multimodal_train.jsonl'
VAL_FILE = OUT_DIR / 'multimodal_val.jsonl'
TEST_FILE = OUT_DIR / 'multimodal_test.jsonl'
SUMMARY = OUT_DIR / 'splits_summary.csv'
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
SEED = 42

def load_records() -> list[dict]:
    print(f'[1/4] Reading {INPUT.name} ...', flush=True)
    if not INPUT.exists():
        sys.exit(f'ERROR: {INPUT} not found. Run 01_build_multimodal_join.py first.')
    records = [json.loads(line) for line in INPUT.open()]
    print(f'      {len(records):>9,d} records', flush=True)
    return records

def group_by_patient(records: list[dict]) -> dict[str, list[dict]]:
    by_pat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_pat[r['subject_id']].append(r)
    return by_pat

def split_patients(patients: list[str]) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(SEED)
    shuffled = sorted(patients)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return (train, val, test)

def write_split(path: Path, records: list[dict]) -> None:
    with path.open('w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    print(f'[2/4] Grouping by subject_id ...', flush=True)
    by_pat = group_by_patient(records)
    patients = list(by_pat.keys())
    print(f'      {len(patients):>9,d} unique subject_ids', flush=True)
    print(f'[3/4] Splitting patients {TRAIN_FRAC:.0%}/{VAL_FRAC:.0%}/{TEST_FRAC:.0%}  (seed={SEED}) ...', flush=True)
    train_pats, val_pats, test_pats = split_patients(patients)
    assert not set(train_pats) & set(val_pats)
    assert not set(train_pats) & set(test_pats)
    assert not set(val_pats) & set(test_pats)
    train_records = [r for p in train_pats for r in by_pat[p]]
    val_records = [r for p in val_pats for r in by_pat[p]]
    test_records = [r for p in test_pats for r in by_pat[p]]
    print(f'      train : {len(train_records):>7,d} records / {len(train_pats):>6,d} patients', flush=True)
    print(f'      val   : {len(val_records):>7,d} records / {len(val_pats):>6,d} patients', flush=True)
    print(f'      test  : {len(test_records):>7,d} records / {len(test_pats):>6,d} patients', flush=True)
    print(f'[4/4] Writing splits → {OUT_DIR}/ ...', flush=True)
    write_split(TRAIN_FILE, train_records)
    write_split(VAL_FILE, val_records)
    write_split(TEST_FILE, test_records)
    summary = pd.DataFrame([{'split': 'train', 'patients': len(train_pats), 'records': len(train_records)}, {'split': 'val', 'patients': len(val_pats), 'records': len(val_records)}, {'split': 'test', 'patients': len(test_pats), 'records': len(test_records)}, {'split': 'TOTAL', 'patients': len(patients), 'records': len(records)}])
    summary.to_csv(SUMMARY, index=False)
    print(f'      → {SUMMARY}', flush=True)
    print()
    print(summary.to_string(index=False))
    print()
    print('✅ Patient-disjoint splits created (no subject_id appears in more than one split).')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
