from __future__ import annotations
import argparse
import json
import logging
import re as _re
import sys
import time
from datetime import datetime
from pathlib import Path
import importlib.util
HERE = Path(__file__).resolve().parent

def _import_local(stem: str, filename: str):
    spec = importlib.util.spec_from_file_location(stem, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod
_dataset_mod = _import_local('multimodal_dataset', '05_multimodal_dataset.py')
_kg_mod = _import_local('kg_retrieval', '07_kg_retrieval.py')
MultimodalDataset = _dataset_mod.MultimodalDataset
KGRetriever = _kg_mod.KGRetriever
SECTIONS_11 = _kg_mod.SECTIONS_11
from models_config import get_model, ModelSpec
logger = logging.getLogger('generate_chexagent')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
SYSTEM_PROMPT = "You are a clinical writing assistant. Given a patient's discharge note, a chest X-ray image, and a paired radiology report, write the BRIEF HOSPITAL COURSE (BHC) section. Use ONLY information present in the note, image, and report. Do NOT introduce facts that are not supported by the sources."

def build_user_text(record: dict, kg_block: str) -> str:
    parts = ['CLINICAL NOTE:', record['input_note'].strip(), '', 'PAIRED RADIOLOGY REPORT:', record['report'].strip()]
    if kg_block:
        parts += ['', kg_block]
    parts += ['', 'Write the Brief Hospital Course (BHC):']
    return '\n'.join(parts)

class _CheXagentComponents:

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def decode(self, token_ids, skip_special_tokens: bool=True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

def _build_chexagent(components: _CheXagentComponents, record: dict, kg_block: str, no_image: bool=False) -> dict:
    user_text = SYSTEM_PROMPT + '\n\n' + build_user_text(record, kg_block)
    prompt = f' USER: <s>{user_text} ASSISTANT: <s>'
    input_ids = components.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    result = {'input_ids': input_ids}
    if not no_image and record.get('image') is not None:
        pixel_values = components.image_processor(record['image'], return_tensors='pt')['pixel_values']
        result['pixel_values'] = pixel_values.unsqueeze(1)
    return result
PROMPT_BUILDERS = {'LLaMA + CXR encoder': _build_chexagent, 'Gemma': lambda *a, **k: (_ for _ in ()).throw(NotImplementedError('Use the MedGemma scripts for MedGemma')), 'Mistral/LLaVA': lambda *a, **k: (_ for _ in ()).throw(NotImplementedError('Use the LLaVA-Med scripts for LLaVA-Med'))}

def _find_best_checkpoint(run_base: Path) -> Path | None:
    if not run_base.exists():
        return None
    for ts_dir in sorted(run_base.iterdir(), reverse=True):
        if not ts_dir.is_dir():
            continue
        ckpts = sorted(ts_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]), reverse=True)
        for ckpt in ckpts:
            state_file = ckpt / 'trainer_state.json'
            if not state_file.exists():
                continue
            state = json.loads(state_file.read_text())
            best = state.get('best_model_checkpoint')
            if best and Path(best).exists():
                return Path(best)
            return ckpt
        if (ts_dir / 'adapter_config.json').exists():
            return ts_dir
    return None

def load_model_and_processor(spec: ModelSpec, adapter_path: Path | None, quant_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    logger.info(f'Loading processor: {spec.hf_id} (trust_remote_code)')
    processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    kwargs: dict = {'torch_dtype': torch.bfloat16, 'device_map': 'auto', 'trust_remote_code': True, 'attn_implementation': 'eager'}
    if quant_4bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    logger.info(f'Loading model: {spec.hf_id}  (4bit={quant_4bit})')
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    _CAC = type(model)
    if not getattr(_CAC, '_cache_pos_patched', False):
        _orig_cls_fwd = _CAC.forward

        def _cls_fwd_no_cache_pos(self, *args, **kwargs):
            kwargs.pop('cache_position', None)
            return _orig_cls_fwd(self, *args, **kwargs)
        _CAC.forward = _cls_fwd_no_cache_pos
        _CAC._cache_pos_patched = True
        logger.info(f'Patched {_CAC.__name__}.forward to drop cache_position')
    if adapter_path is not None:
        from peft import PeftModel
        logger.info(f'Attaching LoRA adapter from {adapter_path}')
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return (model, _CheXagentComponents(tokenizer, processor.image_processor))
_ASSISTANT_LEAK_RE = _re.compile('^(?:ASSISTANT:\\s*<s>|ASSISTANT:|SISTANT:|ISTANT:|STANT:|TANT:|ANT:|NT:|T:)\\s*')

def _strip_assistant_leak(text: str) -> str:
    return _ASSISTANT_LEAK_RE.sub('', text, count=1)

def already_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)['note_id'])
            except Exception:
                pass
    return done

def generate_one(model, processor, builder, record: dict, kg_block: str, max_new_tokens: int, do_sample: bool, repetition_penalty: float=1.0, no_repeat_ngram_size: int=0, min_new_tokens: int=0, no_image: bool=False) -> str:
    import torch
    inputs = builder(processor, record, kg_block, no_image=no_image)
    device = next(model.parameters()).device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = inputs.get('pixel_values')
    if pixel_values is not None:
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    this_min = min(min_new_tokens, max(max_new_tokens - 1, 0))
    gen_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, min_new_tokens=this_min, do_sample=do_sample, temperature=0.2 if do_sample else 1.0, top_p=0.9 if do_sample else 1.0, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
    if pixel_values is not None:
        gen_kwargs['pixel_values'] = pixel_values
    with torch.inference_mode():
        out_ids = model.generate(**gen_kwargs)
    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return _strip_assistant_leak(text)

def generate_visual_summary(model, processor: _CheXagentComponents, image, max_new_tokens: int=180) -> str:
    import torch
    if image is None:
        return ''
    prompt_text = SYSTEM_PROMPT + '\n\nDescribe this chest X-ray in 2-3 concise sentences. Focus on clinically relevant findings only. Do not speculate beyond what is visible in the image.'
    prompt = f' USER: <s>{prompt_text} ASSISTANT: <s>'
    device = next(model.parameters()).device
    input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    pixel_values = processor.image_processor(image, return_tensors='pt')['pixel_values'].to(device=device, dtype=torch.bfloat16).unsqueeze(1)
    with torch.inference_mode():
        out_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return _strip_assistant_leak(text)

def generate_radiology_summary(model, processor: _CheXagentComponents, report_text: str, max_new_tokens: int=180) -> str:
    import torch
    report_text = (report_text or '').strip()
    if not report_text:
        return ''
    prompt_text = SYSTEM_PROMPT + f'\n\nSummarize the following chest X-ray report in 2-3 concise sentences. Include only clinically relevant findings stated in the report.\n\nREPORT:\n{report_text}\n\nSUMMARY:'
    prompt = f' USER: <s>{prompt_text} ASSISTANT: <s>'
    device = next(model.parameters()).device
    input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        out_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return _strip_assistant_leak(text)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='chexagent_8b')
    ap.add_argument('--mode', default=None, choices=['ft', 'ft_mm', 'raft'], help='Auto-find best adapter; sets --no-kg for ft and ft_mm (neither adapter ever saw a KG block in training, so injecting one at inference would be out-of-regime).')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--adapter', default=None, help='Explicit adapter path (overrides --mode).')
    ap.add_argument('--pilot', type=int, default=0, help='Generate only first N records (0 = all).')
    ap.add_argument('--smoke', action='store_true', help='Shorthand for --pilot 3 --split val.')
    ap.add_argument('--start-idx', type=int, default=0)
    ap.add_argument('--end-idx', type=int, default=None)
    ap.add_argument('--max-new-tokens', type=int, default=512)
    ap.add_argument('--min-new-tokens', type=int, default=0)
    ap.add_argument('--adaptive-length', action='store_true')
    ap.add_argument('--include-radiology', action='store_true')
    ap.add_argument('--include-vqa', action='store_true')
    ap.add_argument('--no-image', action='store_true', help="Text-only ablation baseline: omit pixel_values from the main BHC generation call entirely (note+report(+KG) only). Also skips --include-vqa's visual summary pass. --include-radiology still works normally (text-only: report → summary).")
    ap.add_argument('--outdir', type=str, default=None, help='Explicit output directory for generated predictions. Default: auto-named spec.generated_dir path.')
    ap.add_argument('--no-4bit', action='store_true')
    ap.add_argument('--kg-budget-parity', action='store_true', help='Ablation aid: with --no-kg and --adaptive-length, still size the generation budget as if the KG block were present, so the no-KG arm differs from the KG arm only in prompt content and not in tokens allowed. No effect unless both --no-kg and --adaptive-length are set.')
    ap.add_argument('--no-kg', action='store_true')
    ap.add_argument('--sample', action='store_true')
    ap.add_argument('--repetition-penalty', type=float, default=1.0)
    ap.add_argument('--no-repeat-ngram', type=int, default=0)
    ap.add_argument('--resume', type=str, default=None, help='Path to existing partial .jsonl to continue.')
    args = ap.parse_args()
    if args.smoke:
        args.pilot = 3
        args.split = 'val'
        logger.info('[smoke] --smoke → --pilot 3 --split val')
    spec = get_model(args.model)
    if args.mode is not None and args.adapter is None:
        run_base = spec.run_dir / args.mode / 'fsdp'
        best = _find_best_checkpoint(run_base)
        if best:
            args.adapter = str(best)
            logger.info(f'[auto-adapter] mode={args.mode} → {args.adapter}')
        else:
            logger.warning(f'[auto-adapter] no checkpoint found in {run_base}')
        if args.mode in ('ft', 'ft_mm') and (not args.no_kg):
            args.no_kg = True
            logger.info(f'[auto] {args.mode} mode: KG disabled (adapter was trained without a KG block)')
    if spec.family not in PROMPT_BUILDERS:
        sys.exit(f'No prompt builder for family {spec.family!r}.')
    builder = PROMPT_BUILDERS[spec.family]
    ds = MultimodalDataset(args.split)
    ds_len = len(ds)
    hi = ds_len if args.end_idx is None else min(args.end_idx, ds_len)
    lo = max(0, args.start_idx)
    if args.pilot > 0:
        hi = min(hi, lo + args.pilot)
    indices = range(lo, hi)
    n_total = len(indices)
    is_shard = lo != 0 or args.end_idx is not None
    out_dir = Path(args.outdir) if args.outdir else spec.generated_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        out_path = Path(args.resume)
        if not out_path.exists():
            sys.exit(f'--resume path does not exist: {out_path}')
        ts = out_path.stem.split('_')[-1]
        logger.info(f'Resuming into existing file → {out_path}')
    else:
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        mode_tag = f'_{args.mode}' if args.mode else ''
        shard_tag = f'_shard{lo}-{hi}' if is_shard else ''
        out_path = out_dir / f'{args.split}{mode_tag}{shard_tag}_{ts}.jsonl'
        logger.info(f'Writing predictions → {out_path}')
    retriever = KGRetriever()
    model, processor = load_model_and_processor(spec, Path(args.adapter) if args.adapter else None, quant_4bit=not args.no_4bit)
    logger.info(f'Generating on {n_total:,} records (split={args.split}, indices [{lo}, {hi}))')
    done = already_done_ids(out_path)
    if done:
        logger.info(f'Resuming: {len(done):,} records already in output.')
    t0 = time.time()
    n_written = 0
    with out_path.open('a') as fout:
        for i in indices:
            rec = ds[i]
            if rec['note_id'] in done:
                continue
            _kg_full = ''
            if not args.no_kg or args.kg_budget_parity:
                _kg_full = retriever.build_kg_block(note_text=rec['input_note'], dicom_id=str(rec.get('dicom_id') or Path(rec['cxr_image_path']).stem), section='hospital_course')
            kg_block = '' if args.no_kg else _kg_full
            if args.adaptive_length:
                note_chars = len(rec.get('input_note', ''))
                report_chars = len(rec.get('radiology_report', '') or rec.get('report', ''))
                kg_chars = len(_kg_full) if args.kg_budget_parity else len(kg_block)
                evidence_tokens = (note_chars + report_chars + kg_chars) // 4
                this_max_new = max(384, min(1024, int(evidence_tokens * 0.35)))
            else:
                this_max_new = args.max_new_tokens
            try:
                pred = generate_one(model, processor, builder, rec, kg_block, max_new_tokens=this_max_new, min_new_tokens=args.min_new_tokens, do_sample=args.sample, repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram, no_image=args.no_image)
                if args.include_radiology:
                    rad_report = (rec.get('radiology_report') or rec.get('report') or '').strip()
                    if rad_report:
                        try:
                            rad_summary = generate_radiology_summary(model, processor, rad_report, max_new_tokens=180)
                            if rad_summary:
                                pred = pred.rstrip() + '\n\n## Radiology Findings\n' + rad_summary
                        except Exception as e:
                            logger.warning(f'[{i}] radiology summary failed: {e}')
                if args.include_vqa and (not args.no_image):
                    img = rec.get('image')
                    if img is not None:
                        try:
                            vqa_summary = generate_visual_summary(model, processor, img, max_new_tokens=180)
                            if vqa_summary:
                                pred = pred.rstrip() + '\n\n## Visual Interpretation of CXR (model)\n' + vqa_summary
                        except Exception as e:
                            logger.warning(f'[{i}] VQA summary failed: {e}')
            except Exception as e:
                logger.error(f"[{i}] note_id={rec['note_id']}: {e}")
                continue
            fout.write(json.dumps({'note_id': rec['note_id'], 'subject_id': rec['subject_id'], 'study_id': rec['study_id'], 'cxr_image_path': rec.get('cxr_image_path', ''), 'dicom_id': rec.get('dicom_id', ''), 'prediction': pred, 'target': rec['target'], 'kg_used': bool(kg_block)}) + '\n')
            fout.flush()
            n_written += 1
            if args.smoke:
                print(f"\n{'=' * 80}")
                print(f"RECORD {i + 1}/3  note_id={rec['note_id']}")
                print(f"{'=' * 80}")
                print('TARGET (first 300 chars):')
                print(rec['target'][:300])
                print('\nPREDICTION:')
                print(pred[:800])
                print(f"{'=' * 80}\n")
            if n_written % 25 == 0 and (not args.smoke):
                rate = n_written / (time.time() - t0)
                logger.info(f'  {n_written:>5d} / {n_total}  ({rate:.2f} rec/s)')
    elapsed = time.time() - t0
    logger.info(f'✅ Done. {n_written:,} predictions in {elapsed:.1f}s')
    json_path = out_path.with_suffix('.json')
    txt_path = out_path.with_suffix('.txt')
    all_records: list[dict] = []
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    with json_path.open('w') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    import textwrap

    def _wrap(text: str, width: int=100) -> str:
        paras = (text or '').strip().split('\n')
        wrapped = []
        for p in paras:
            if not p.strip():
                wrapped.append('')
            else:
                wrapped.append(textwrap.fill(p, width=width, break_long_words=False, break_on_hyphens=False))
        return '\n'.join(wrapped)
    with txt_path.open('w') as f:
        for i, r in enumerate(all_records, 1):
            f.write('=' * 100 + '\n')
            f.write(f'RECORD {i} / {len(all_records)}\n')
            f.write(f"note_id:    {r.get('note_id', '')}\n")
            f.write(f"study_id:   {r.get('study_id', '')}\n")
            f.write(f"kg_used:    {r.get('kg_used', '')}\n")
            f.write('=' * 100 + '\n\n')
            f.write('--- REFERENCE BHC ---\n')
            f.write(_wrap(r.get('target', '')) + '\n\n')
            f.write('--- GENERATED BHC ---\n')
            f.write(_wrap(r.get('prediction', '')) + '\n\n\n')
    logger.info(f'  also wrote: {json_path.name} and {txt_path.name}')
    try:
        import importlib.util as _ilu
        cleaner_py = HERE / 'clean_predictions.py'
        _spec = _ilu.spec_from_file_location('_cleaner', str(cleaner_py))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        cleaned_records = []
        for r in all_records:
            out = dict(r)
            out['prediction_raw'] = r.get('prediction', '')
            out['prediction'] = _mod.clean(r.get('prediction', ''))
            out['target_raw'] = r.get('target', '')
            out['target'] = _mod.clean(r.get('target', ''))
            cleaned_records.append(out)
        cl_jsonl = out_path.with_suffix('.cleaned.jsonl')
        cl_json = out_path.with_suffix('.cleaned.json')
        cl_txt = out_path.with_suffix('.cleaned.txt')
        with cl_jsonl.open('w') as f:
            for r in cleaned_records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        with cl_json.open('w') as f:
            json.dump(cleaned_records, f, indent=2, ensure_ascii=False)
        with cl_txt.open('w') as f:
            for i, r in enumerate(cleaned_records, 1):
                f.write('=' * 100 + '\n')
                f.write(f'RECORD {i} / {len(cleaned_records)}\n')
                f.write(f"note_id:    {r.get('note_id', '')}\n")
                f.write(f"study_id:   {r.get('study_id', '')}\n")
                f.write(f"kg_used:    {r.get('kg_used', '')}\n")
                f.write('=' * 100 + '\n\n')
                f.write('--- REFERENCE BHC (cleaned) ---\n')
                f.write(_wrap(r.get('target', '')) + '\n\n')
                f.write('--- GENERATED BHC (Generated Summary) ---\n')
                f.write(_wrap(r.get('prediction', '')) + '\n\n\n')
        logger.info(f'  also wrote cleaned: {cl_jsonl.name} / {cl_json.name} / {cl_txt.name}')
    except Exception as _e:
        logger.warning(f'Auto-clean failed ({_e}); raw .jsonl is unaffected.')
    manifest = {'model_short_name': spec.short_name, 'model_hf_id': spec.hf_id, 'split': args.split, 'mode': args.mode, 'adapter': args.adapter, 'pilot': args.pilot, 'smoke': args.smoke, 'no_kg': args.no_kg, 'kg_budget_parity': args.kg_budget_parity, 'no_4bit': args.no_4bit, 'no_image': args.no_image, 'include_radiology': args.include_radiology, 'include_vqa': args.include_vqa, 'sample': args.sample, 'max_new_tokens': args.max_new_tokens, 'min_new_tokens': args.min_new_tokens, 'adaptive_length': args.adaptive_length, 'n_written': n_written, 'elapsed_seconds': round(elapsed, 1), 'output_jsonl': str(out_path), 'timestamp': ts}
    (out_dir / f'{out_path.stem}_manifest.json').write_text(json.dumps(manifest, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
