from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any
try:
    from PIL import Image
except ImportError:
    sys.exit('ERROR: Pillow not installed. Run: pip install Pillow')
try:
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    Dataset = object
    _TORCH_AVAILABLE = False
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'
_IMAGE_DIR = DATA_DIR / 'preprocessed' / 'images'

def _split_file(split: str) -> Path:
    for suffix in (f'multimodal_{split}_clean.jsonl', f'multimodal_{split}_cleaned.jsonl', f'multimodal_{split}_preprocessed.jsonl', f'multimodal_{split}.jsonl'):
        p = DATA_DIR / suffix
        if p.exists():
            return p
    return DATA_DIR / f'multimodal_{split}_preprocessed.jsonl'

def _remap_image_path(stored: str) -> Path:
    stem = Path(stored).stem
    local = _IMAGE_DIR / f'{stem}.jpg'
    if local.exists():
        return local
    return Path(stored)

class MultimodalDataset(Dataset):

    def __init__(self, split: str, transform_image=None, keep_in_memory: bool=True) -> None:
        path = _split_file(split)
        if not path.exists():
            raise FileNotFoundError(f'{path} not found. Run 04_preprocess.py first.')
        self.split = split
        self.path = path
        self.transform_image = transform_image
        self.keep_in_memory = keep_in_memory
        if keep_in_memory:
            self._records: list[dict] = [json.loads(line) for line in path.open()]
        else:
            self._offsets: list[int] = []
            with path.open('rb') as f:
                pos = 0
                for line in f:
                    self._offsets.append(pos)
                    pos += len(line)

    def __len__(self) -> int:
        return len(self._records) if self.keep_in_memory else len(self._offsets)

    def _read_record(self, idx: int) -> dict:
        if self.keep_in_memory:
            return self._records[idx]
        with self.path.open('rb') as f:
            f.seek(self._offsets[idx])
            return json.loads(f.readline().decode('utf-8'))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._read_record(idx)
        img_path = _remap_image_path(r['cxr_image_path'])
        img = Image.open(img_path).convert('RGB')
        if self.transform_image is not None:
            img = self.transform_image(img)
        return {'subject_id': r['subject_id'], 'study_id': r['study_id'], 'dicom_id': Path(r['cxr_image_path']).stem, 'note_id': r['note_id'], 'input_note': r['input_note'], 'image': img, 'cxr_image_path': str(img_path), 'report': r['radiology_report'], 'target': r['target_bhc']}

def _show(n: int) -> None:
    for split in ('train', 'val', 'test'):
        ds = MultimodalDataset(split)
        print(f'\n=== {split}  (n = {len(ds):,}) ===')
        for i in range(min(n, len(ds))):
            s = ds[i]
            print(f"  [{i}] subject={s['subject_id']}  study={s['study_id']}")
            print(f"      image: {s['image'].size}  mode={s['image'].mode}")
            print(f"      note ({len(s['input_note'])} ch): {s['input_note'][:120]}...")
            print(f"      tgt  ({len(s['target'])} ch): {s['target'][:120]}...")
            print(f"      rpt  ({len(s['report'])} ch): {s['report'][:120]}...")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--show', type=int, default=0, help='Print first N records from each split.')
    args = ap.parse_args()
    sizes = {}
    for split in ('train', 'val', 'test'):
        ds = MultimodalDataset(split)
        sizes[split] = len(ds)
        _ = ds[0]
    print(f"train: {sizes['train']:,}  val: {sizes['val']:,}  test: {sizes['test']:,}  (total: {sum(sizes.values()):,})")
    if args.show:
        _show(args.show)
    if _TORCH_AVAILABLE:
        from torch.utils.data import DataLoader
        ds = MultimodalDataset('val')
        dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=lambda batch: batch)
        b = next(iter(dl))
        print(f"\nDataLoader smoke-test: batch size = {len(b)}, first subject_id = {b[0]['subject_id']}")
    else:
        print('\n(torch not installed — DataLoader smoke-test skipped)')
    print('\n✅ Step 5 dataset class verified.  Ready for per-model collator (step 6).')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
