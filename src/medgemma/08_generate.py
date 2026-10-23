from __future__ import annotations
import os as _os
if not any((k in _os.environ for k in ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'))):
    _os.environ['ROCR_VISIBLE_DEVICES'] = '0,1'
    _os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
    _os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import argparse
import json
import logging
import sys
import time
import traceback
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
logger = logging.getLogger('generate')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
SYSTEM_PROMPT = "You are a clinical writing assistant. Given a patient's discharge note, a chest X-ray image, and a paired radiology report, write the BRIEF HOSPITAL COURSE (BHC) section. Use ONLY information present in the note, image, and report. Do NOT introduce facts that are not supported by the sources."

def build_user_text(record: dict, kg_block: str) -> str:
    parts = ['CLINICAL NOTE:', record['input_note'].strip(), '', 'PAIRED RADIOLOGY REPORT:', record['report'].strip()]
    if kg_block:
        parts += ['', kg_block]
    parts += ['', 'Write the Brief Hospital Course (BHC):']
    return '\n'.join(parts)

def _build_medgemma(processor, record: dict, kg_block: str, no_image: bool=False):
    user_text = build_user_text(record, kg_block)
    user_content = [{'type': 'text', 'text': user_text}]
    if not no_image:
        user_content.insert(0, {'type': 'image', 'image': record['image']})
    messages = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]}, {'role': 'user', 'content': user_content}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')

def _not_implemented(family: str):

    def _f(*args, **kwargs):
        raise NotImplementedError(f'Prompt builder for family={family!r} is not implemented yet. Add a `_build_<family>` function modelled on `_build_medgemma`.')
    return _f
PROMPT_BUILDERS = {'Gemma': _build_medgemma, 'Gemma 2 + SigLIP': _not_implemented('PaliGemma 2'), 'Mistral/LLaVA': _not_implemented('LLaVA-Med'), 'LLaMA + CXR encoder': _not_implemented('CheXagent-2'), 'Qwen3': _not_implemented('Qwen3-VL'), 'Phi-4': _not_implemented('Phi-4-MM'), 'InternLM + InternViT': _not_implemented('InternVL 2.5'), 'LLaMA + image encoder': _not_implemented('MAIRA-2')}

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
            state = json.load(state_file.open())
            best = state.get('best_model_checkpoint')
            if best and Path(best).exists():
                return Path(best)
            return ckpt
    return None

def _read_safetensors_header_metadata(path: str) -> dict:
    import struct, json as _json
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = _json.loads(f.read(header_len).decode('utf-8'))
    return header.get('__metadata__', {})

def _patch_no_mmap_safe_open():
    import struct as _struct
    import json as _json_mod
    import torch as _torch
    _ST_DTYPE_MAP = {'F64': _torch.float64, 'F32': _torch.float32, 'F16': _torch.float16, 'BF16': _torch.bfloat16, 'I64': _torch.int64, 'I32': _torch.int32, 'I16': _torch.int16, 'I8': _torch.int8, 'U8': _torch.uint8, 'BOOL': _torch.bool}
    _DTYPE_BYTES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}

    def _read_tensor(path: str, data_start: int, dtype_str: str, shape: list, start: int, end: int, device: str) -> '_torch.Tensor':
        n_bytes = end - start
        with open(path, 'rb') as fh:
            fh.seek(data_start + start)
            raw = fh.read(n_bytes)
        dtype = _ST_DTYPE_MAP[dtype_str]
        if dtype == _torch.bfloat16:
            t = _torch.frombuffer(bytearray(raw), dtype=_torch.int16).view(_torch.bfloat16)
        else:
            t = _torch.frombuffer(bytearray(raw), dtype=dtype)
        t = t.reshape(shape)
        if device != 'cpu':
            t = t.to(device)
        return t.clone()

    class _LazySlice:

        def __init__(self, path, data_start, dtype_str, shape, start, end, device):
            self._path = path
            self._data_start = data_start
            self._dtype_str = dtype_str
            self._shape = shape
            self._start = start
            self._end = end
            self._device = device
            self._tensor = None

        def get_dtype(self) -> str:
            return self._dtype_str

        def get_shape(self) -> list:
            return list(self._shape)

        def __getitem__(self, idx):
            if self._tensor is None:
                self._tensor = _read_tensor(self._path, self._data_start, self._dtype_str, self._shape, self._start, self._end, self._device)
            return self._tensor[idx]

    class _LustreSafeOpen:

        def __init__(self, path, framework='pt', device='cpu'):
            self._path = path
            self._device = device
            with open(path, 'rb') as fh:
                hdr_len = _struct.unpack('<Q', fh.read(8))[0]
                hdr = _json_mod.loads(fh.read(hdr_len))
                self._data_start = 8 + hdr_len
            self._meta = hdr.get('__metadata__', {'format': 'pt'})
            self._index = {k: v for k, v in hdr.items() if k != '__metadata__'}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def metadata(self):
            return self._meta

        def keys(self):
            return list(self._index.keys())

        def get_slice(self, key) -> _LazySlice:
            info = self._index[key]
            return _LazySlice(self._path, self._data_start, info['dtype'], info['shape'], info['data_offsets'][0], info['data_offsets'][1], self._device)

        def get_tensor(self, key) -> '_torch.Tensor':
            info = self._index[key]
            return _read_tensor(self._path, self._data_start, info['dtype'], info['shape'], info['data_offsets'][0], info['data_offsets'][1], self._device)

        def get_tensors(self) -> dict:
            return {k: self.get_tensor(k) for k in self._index}
    import transformers.modeling_utils as _tmu
    _tmu.safe_open = _LustreSafeOpen
    try:
        import transformers.safetensors_utils as _stu
        _stu.safe_open = _LustreSafeOpen
    except ImportError:
        pass
    logger.info('Patched safetensors safe_open → Lustre-safe header-only + lazy read')

def load_model_and_processor(spec: ModelSpec, adapter_path: Path | None, quant_4bit: bool):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    logger.info(f'Loading processor: {spec.hf_id}')
    processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
    kwargs = {'torch_dtype': torch.bfloat16, 'device_map': 'auto', 'trust_remote_code': True, 'low_cpu_mem_usage': True}
    if quant_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        except Exception as e:
            logger.warning(f'bitsandbytes 4-bit unavailable ({e}); using bf16.')
    _patch_no_mmap_safe_open()
    logger.info(f'Loading model: {spec.hf_id}  (4bit={quant_4bit})')
    model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, **kwargs)
    cfg = getattr(model, 'config', None)
    if cfg is not None and hasattr(cfg, 'cache_implementation'):
        cfg.cache_implementation = None
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as _AAF
        if not getattr(_AAF, '_swa_kv_patched', False):
            _orig_sdpa_fwd = _AAF['sdpa']

            def _swa_sdpa_fwd(module, query, key, value, attention_mask, **kw):
                if attention_mask is not None and attention_mask.shape[-1] < key.shape[-2]:
                    w = attention_mask.shape[-1]
                    key = key[:, :, -w:, :]
                    value = value[:, :, -w:, :]
                return _orig_sdpa_fwd(module, query, key, value, attention_mask, **kw)
            _AAF['sdpa'] = _swa_sdpa_fwd
            _AAF._swa_kv_patched = True
    except Exception as _e:
        logger.warning(f'SDPA SWA-KV patch failed: {_e}')
    logger.info('Set cache_implementation=None + patched ALL_ATTENTION_FUNCTIONS[sdpa] for SWA KV slice')
    try:
        from transformers.cache_utils import Cache as _Cache
        if not hasattr(_Cache, 'float'):
            _Cache.float = lambda self: self
        if not hasattr(_Cache, 'to'):
            _Cache.to = lambda self, *a, **kw: self
    except ImportError:
        pass
    if adapter_path is not None:
        from peft import PeftModel
        logger.info(f'Attaching LoRA adapter from {adapter_path}')
        model = PeftModel.from_pretrained(model, str(adapter_path))
        base_cfg = getattr(getattr(model, 'base_model', None), 'config', None)
        if base_cfg is not None and hasattr(base_cfg, 'cache_implementation'):
            base_cfg.cache_implementation = None
    model.eval()
    return (model, processor)

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

def _first_device(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_one(model, processor, builder, record: dict, kg_block: str, max_new_tokens: int, do_sample: bool, repetition_penalty: float=1.0, no_repeat_ngram_size: int=0, min_new_tokens: int=0, no_image: bool=False) -> str:
    import torch
    inputs = builder(processor, record, kg_block, no_image=no_image)
    dev = _first_device(model)
    inputs = {k: v.to(dev) if hasattr(v, 'to') else v for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[-1]
    this_min_new = min(min_new_tokens, max(max_new_tokens - 1, 0))
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=this_min_new, do_sample=do_sample, temperature=0.2 if do_sample else 1.0, top_p=0.9 if do_sample else 1.0, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size)
    gen_ids = out_ids[0, prompt_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

def generate_visual_summary(model, processor, image, max_new_tokens: int=180) -> str:
    import torch
    if image is None:
        return ''
    prompt = 'Describe this chest X-ray in 2-3 concise sentences. Focus on clinically relevant findings only. Do not speculate beyond what is visible in the image.'
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True, tokenize=True)
    dev = _first_device(model)
    inputs = {k: v.to(dev) if hasattr(v, 'to') else v for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[-1]
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
    gen_ids = out_ids[0, prompt_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

def generate_radiology_summary(model, processor, report_text: str, max_new_tokens: int=180) -> str:
    import torch
    report_text = (report_text or '').strip()
    if not report_text:
        return ''
    prompt = f'Summarize the following chest X-ray report in 2-3 concise sentences. Include only clinically relevant findings stated in the report (no speculation, no findings not present in the report).\n\nREPORT:\n{report_text}\n\nSUMMARY:'
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True, tokenize=True)
    dev = _first_device(model)
    inputs = {k: v.to(dev) if hasattr(v, 'to') else v for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[-1]
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0)
    gen_ids = out_ids[0, prompt_len:]
    return processor.decode(gen_ids, skip_special_tokens=True).strip()

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='medgemma_27b', help='Short name from models_config.py')
    ap.add_argument('--mode', default=None, choices=['ft', 'ft_mm', 'raft'], help='Training mode. Auto-finds the best checkpoint and sets --no-kg correctly. ft and ft_mm were both trained WITHOUT a KG block, so injecting one at inference would be out-of-regime; raft trained with it.')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--adapter', default=None, help='Path to a LoRA adapter dir (optional; overrides --mode)')
    ap.add_argument('--pilot', type=int, default=0, help='If >0, generate only the first N records (smoke test)')
    ap.add_argument('--start-idx', type=int, default=0, help='Shard start index into the split (inclusive). Combine with --end-idx to run disjoint shards on separate nodes in parallel.')
    ap.add_argument('--end-idx', type=int, default=None, help='Shard end index into the split (exclusive). Default = len(dataset) (or --pilot cap if set).')
    ap.add_argument('--max-new-tokens', type=int, default=512, help='Fixed token cap on generation. Overridden per-record if --adaptive-length is set.')
    ap.add_argument('--adaptive-length', action='store_true', help='Set max_new_tokens per record based on the total evidence length (note + report + KG block). Formula: max_new = clamp(evidence_tokens * 0.35, 384, 1024). Avoids hallucination on simple cases and truncation on complex cases.')
    ap.add_argument('--min-new-tokens', type=int, default=0, help='Force generation to continue past EOS until at least this many BHC tokens are produced (0 = off). Reference BHCs run ~300-600+ words; without a floor the model sometimes stops after ~50-100 words on cases it treats as simple. Try 200-250.')
    ap.add_argument('--include-radiology', action='store_true', help="After generating the BHC, do a second-pass generation that summarizes the radiology report into 2-3 sentences and appends as a '## Radiology Findings' section. Increases inference time ~50%%.")
    ap.add_argument('--include-vqa', action='store_true', help="After generating the BHC, do a vision-only second pass asking the model to describe the CXR image itself in 2-3 sentences. Appends as '## Visual Interpretation of CXR (model)'. Stronger multimodal claim than --include-radiology (which only reads the report text). Increases inference time ~50%%.")
    ap.add_argument('--no-image', action='store_true', help="Text-only ablation baseline: drop the CXR pixels from the main BHC generation call entirely (note+report(+KG) only). Also skips --include-vqa's visual summary pass since there's no image to describe. --include-radiology still works normally (it's text-only: report → summary).")
    ap.add_argument('--outdir', type=str, default=None, help='Explicit output directory for generated predictions (e.g. runs/medgemma_27b/generated/ft_1000_baseline). Default: auto-named spec.generated_dir path.')
    ap.add_argument('--no-4bit', action='store_true', help='Disable 4-bit NF4 quantisation (uses bf16 only)')
    ap.add_argument('--kg-budget-parity', action='store_true', help='Ablation aid: with --no-kg and --adaptive-length, still size the generation budget as if the KG block were present, so the no-KG arm differs from the KG arm only in prompt content and not in tokens allowed. Costs the same KG-building time as a normal RAFT run. No effect unless both --no-kg and --adaptive-length are set.')
    ap.add_argument('--no-kg', action='store_true', help='Skip KG block (ablation)')
    ap.add_argument('--sample', action='store_true', help='Use sampling (temp=0.2, top_p=0.9). Default = greedy.')
    ap.add_argument('--repetition-penalty', type=float, default=1.0, help="Repetition penalty (1.0 = off). Try 1.15-1.3 to break greedy decoding loops (e.g., 'metoprolol metoprolol...').")
    ap.add_argument('--no-repeat-ngram', type=int, default=0, help='Block exact n-gram repeats during generation (0 = off). Try 5 to forbid any 5-token sequence from appearing twice — strongest anti-loop guard.')
    ap.add_argument('--resume', type=str, default=None, help='Path to an existing (partial) output .jsonl from a prior run of this exact command. Reuses that file instead of creating a new timestamped one, and already_done_ids() skips note_ids already present — so a crashed/killed run can continue without redoing finished records.')
    args = ap.parse_args()
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
        sys.exit(f'No prompt builder registered for family {spec.family!r}.')
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
        shard_suffix = f'_shard{lo}-{hi}' if is_shard else ''
        out_path = out_dir / f'{args.split}{mode_tag}{shard_suffix}_{ts}.jsonl'
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
                _kg_full = retriever.build_kg_block(note_text=rec['input_note'], dicom_id=str(rec.get('dicom_id') or __import__('pathlib').Path(rec['cxr_image_path']).stem), section='hospital_course')
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
                pred = generate_one(model, processor, builder, rec, kg_block, max_new_tokens=this_max_new, do_sample=args.sample, repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram, min_new_tokens=args.min_new_tokens, no_image=args.no_image)
                if args.include_radiology:
                    rad_report = (rec.get('radiology_report') or rec.get('report') or '').strip()
                    if rad_report:
                        try:
                            rad_summary = generate_radiology_summary(model, processor, rad_report, max_new_tokens=180)
                            if rad_summary:
                                pred = pred.rstrip() + '\n\n## Radiology Findings\n' + rad_summary
                        except Exception as e:
                            logger.warning(f"[{i}] radiology summary failed for {rec['note_id']}: {e}")
                if args.include_vqa and (not args.no_image):
                    img = rec.get('image')
                    if img is not None:
                        try:
                            vqa_summary = generate_visual_summary(model, processor, img, max_new_tokens=180)
                            if vqa_summary:
                                pred = pred.rstrip() + '\n\n## Visual Interpretation of CXR (model)\n' + vqa_summary
                        except Exception as e:
                            logger.warning(f"[{i}] VQA summary failed for {rec['note_id']}: {e}")
            except Exception as e:
                logger.error(f"[{i}] note_id={rec['note_id']}: {e}\n{traceback.format_exc()}")
                continue
            fout.write(json.dumps({'note_id': rec['note_id'], 'subject_id': rec['subject_id'], 'study_id': rec['study_id'], 'cxr_image_path': rec.get('cxr_image_path', ''), 'dicom_id': rec.get('dicom_id', ''), 'prediction': pred, 'target': rec['target'], 'kg_used': bool(kg_block)}) + '\n')
            fout.flush()
            n_written += 1
            if n_written % 25 == 0:
                rate = n_written / (time.time() - t0)
                logger.info(f'  {n_written:>5d} / {n_total} done  ({rate:.2f} rec/s)')
    elapsed = time.time() - t0
    logger.info(f'✅ Done. {n_written:,} new predictions in {elapsed:.1f}s ({n_written / max(elapsed, 1e-09):.2f} rec/s)')
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
            f.write(f"note_id:        {r.get('note_id', '')}\n")
            f.write(f"subject_id:     {r.get('subject_id', '')}\n")
            f.write(f"study_id:       {r.get('study_id', '')}\n")
            f.write(f"cxr_image_path: {r.get('cxr_image_path', '')}\n")
            f.write(f"dicom_id:       {r.get('dicom_id', '')}\n")
            f.write(f"kg_used:        {r.get('kg_used', '')}\n")
            f.write('=' * 100 + '\n\n')
            f.write('--- REFERENCE BHC (target) ---\n')
            f.write(_wrap(r.get('target', '')) + '\n\n')
            f.write('--- GENERATED BHC (Generated Summary) ---\n')
            f.write(_wrap(r.get('prediction', '')) + '\n\n\n')
    logger.info(f'  also wrote: {json_path.name} and {txt_path.name}')
    try:
        import importlib.util as _ilu
        cleaner_py = Path(__file__).resolve().parent / 'clean_predictions.py'
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
                f.write(f"note_id:        {r.get('note_id', '')}\n")
                f.write(f"subject_id:     {r.get('subject_id', '')}\n")
                f.write(f"study_id:       {r.get('study_id', '')}\n")
                f.write(f"cxr_image_path: {r.get('cxr_image_path', '')}\n")
                f.write(f"dicom_id:       {r.get('dicom_id', '')}\n")
                f.write(f"kg_used:        {r.get('kg_used', '')}\n")
                f.write('=' * 100 + '\n\n')
                f.write('--- REFERENCE BHC ---\n')
                f.write(_wrap(r['target']) + '\n\n')
                f.write('--- GENERATED BHC (Generated Summary) ---\n')
                f.write(_wrap(r['prediction']) + '\n\n\n')
        logger.info(f'  also wrote cleaned: {cl_jsonl.name} / {cl_json.name} / {cl_txt.name}')
    except Exception as _e:
        logger.warning(f'Auto-clean failed ({_e}); raw .jsonl is unaffected.')
    manifest = {'model_short_name': spec.short_name, 'model_hf_id': spec.hf_id, 'split': args.split, 'adapter': args.adapter, 'pilot': args.pilot, 'no_kg': args.no_kg, 'kg_budget_parity': args.kg_budget_parity, 'no_4bit': args.no_4bit, 'no_image': args.no_image, 'include_radiology': args.include_radiology, 'include_vqa': args.include_vqa, 'sample': args.sample, 'max_new_tokens': args.max_new_tokens, 'min_new_tokens': args.min_new_tokens, 'adaptive_length': args.adaptive_length, 'n_written': n_written, 'elapsed_seconds': round(elapsed, 1), 'output_jsonl': str(out_path), 'output_json': str(json_path), 'output_txt': str(txt_path), 'timestamp': ts}
    (out_dir / f'{out_path.stem}_manifest.json').write_text(json.dumps(manifest, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
