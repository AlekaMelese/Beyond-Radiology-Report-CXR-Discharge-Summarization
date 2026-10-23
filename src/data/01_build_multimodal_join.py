from __future__ import annotations
import json
import sys
import os
from pathlib import Path
import pandas as pd
DATA_DIR = Path(os.environ.get('DATA_DIR', Path(__file__).resolve().parents[2] / 'data'))
MIMIC_CSV = DATA_DIR / 'mimic.csv'
REPORTS_DIR = DATA_DIR / 'mimic-cxr-reports'
JPGS_DIR = DATA_DIR / 'physionet.org' / '2.1.0' / 'files'
OUTPUT = DATA_DIR / 'multimodal_records.jsonl'
SUMMARY_CSV = DATA_DIR / 'multimodal_records_summary.csv'
MAX_NOTES_PER_PATIENT = 1

def load_notes() -> dict[str, dict]:
    print(f'[1/4] Loading {MIMIC_CSV.name} ...', flush=True)
    df = pd.read_csv(MIMIC_CSV, dtype=str)
    df['subject_id'] = df['note_id'].str.split('-', n=1).str[0]
    if MAX_NOTES_PER_PATIENT == 1:
        df = df.drop_duplicates(subset='subject_id', keep='first')
    else:
        df = df.groupby('subject_id').head(MAX_NOTES_PER_PATIENT)
    notes = df.set_index('subject_id').to_dict(orient='index')
    print(f'      {len(notes):>9,d} unique subjects with text notes', flush=True)
    return notes

def index_jpgs() -> list[tuple[str, str, str]]:
    print(f'[2/4] Indexing JPGs under {JPGS_DIR} ...', flush=True)
    out: list[tuple[str, str, str]] = []
    for jpg in JPGS_DIR.rglob('*.jpg'):
        parts = jpg.parts
        try:
            subj = next((p for p in parts if p.startswith('p') and p[1:].isdigit() and (len(p) > 3)))
            stud = next((p for p in parts if p.startswith('s') and p[1:].isdigit()))
        except StopIteration:
            continue
        out.append((subj[1:], stud[1:], str(jpg)))
    print(f'      {len(out):>9,d} JPGs indexed', flush=True)
    return out

def report_path_for(subject_id: str, study_id: str) -> Path:
    return REPORTS_DIR / f'p{subject_id[:2]}' / f'p{subject_id}' / f's{study_id}.txt'

def build_records(notes: dict[str, dict], jpgs: list[tuple[str, str, str]]) -> tuple[int, int, int]:
    print(f'[3/4] Joining → {OUTPUT.name} ...', flush=True)
    n_written = 0
    n_no_note = 0
    n_no_report = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('w') as fout:
        for subject_id, study_id, jpg_path in jpgs:
            note = notes.get(subject_id)
            if note is None:
                n_no_note += 1
                continue
            rpt = report_path_for(subject_id, study_id)
            if not rpt.exists():
                n_no_report += 1
                continue
            try:
                report_text = rpt.read_text(errors='ignore').strip()
            except OSError:
                n_no_report += 1
                continue
            record = {'subject_id': subject_id, 'study_id': study_id, 'note_id': note.get('note_id'), 'input_note': note.get('input'), 'target_bhc': note.get('target'), 'cxr_image_path': jpg_path, 'radiology_report': report_text, 'input_tokens': note.get('input_tokens'), 'target_tokens': note.get('target_tokens')}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            n_written += 1
    print(f'      ✓ Wrote {n_written:>9,d} multimodal records', flush=True)
    print(f'      ✗ Skipped {n_no_note:>9,d} JPGs (no matching note in mimic.csv)', flush=True)
    print(f'      ✗ Skipped {n_no_report:>9,d} JPGs (no matching report on disk)', flush=True)
    print(f'      → {OUTPUT}', flush=True)
    return (n_written, n_no_note, n_no_report)

def write_summary(notes: dict[str, dict], jpgs: list[tuple[str, str, str]], n_written: int, n_no_note: int, n_no_report: int) -> None:
    print(f'[4/4] Writing summary → {SUMMARY_CSV.name} ...', flush=True)
    n_patients_with_note = len(notes)
    n_jpgs_total = len(jpgs)
    n_patients_with_jpg = len({s for s, _, _ in jpgs})
    n_patients_joined = len({s for s, _, _ in jpgs if s in notes})
    summary = pd.DataFrame([{'patients_with_note_mimic_csv': n_patients_with_note, 'jpgs_on_disk': n_jpgs_total, 'patients_with_jpg': n_patients_with_jpg, 'patients_in_join_note_and_jpg': n_patients_joined, 'records_written': n_written, 'skipped_no_note': n_no_note, 'skipped_no_report': n_no_report}])
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f'      → {SUMMARY_CSV}', flush=True)
    print()
    print('Summary:')
    print(summary.T.to_string(header=False))

def main() -> int:
    for p in (MIMIC_CSV, REPORTS_DIR, JPGS_DIR):
        if not p.exists():
            sys.exit(f'ERROR: {p} not found')
    notes = load_notes()
    jpgs = index_jpgs()
    n_written, n_no_note, n_no_report = build_records(notes, jpgs)
    write_summary(notes, jpgs, n_written, n_no_note, n_no_report)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
