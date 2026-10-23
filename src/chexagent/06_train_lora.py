from __future__ import annotations
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from models_config import get_model
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'data'

def _import_local(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
_kg_mod = _import_local('kg_retrieval', '07_kg_retrieval.py')
KGRetriever = _kg_mod.KGRetriever
COMMON_ABBREVS = _kg_mod.COMMON_ABBREVS

def _patch_no_mmap_safe_open():
    import struct as _struct
    import json as _json_mod
    import torch as _torch
    _ST_DTYPE_MAP = {'F64': _torch.float64, 'F32': _torch.float32, 'F16': _torch.float16, 'BF16': _torch.bfloat16, 'I64': _torch.int64, 'I32': _torch.int32, 'I16': _torch.int16, 'I8': _torch.int8, 'U8': _torch.uint8, 'BOOL': _torch.bool}

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
    print('[fix] Patched safetensors safe_open → Lustre-safe header-only + lazy read')
DEFAULT_MODEL = 'chexagent_8b'
LORA_CONFIG = dict(r=16, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
SYSTEM_PROMPT = "You are a clinical summarizer. Generate the Brief Hospital Course (BHC) narrative from the patient's clinical note, chest X-ray, and paired radiology report.  The narrative should cover, in order: (1) Patient and Service; (2) Chief Complaint / Admission Context; (3) History of Present Illness; (4) Past Medical / Surgical History; (5) Medications on Admission; (6) Medications (Discharge / Ongoing); (7) Physical Examination; (8) Investigations / Labs / Imaging; (9) Assessment / Impression; (10) Hospital Course; (11) Discharge Condition; (12) Radiology Findings; (13) Radiology Impression; (14) Procedures Performed; (15) Discharge Disposition; (16) Follow-Up & Recommendations — composed of conditional subsections rendered ONLY when the input supports them: (16.1) Consultations; (16.2) Safety Flags (drug interactions, allergies, contraindications); (16.3) Medication Changes; (16.4) Post-operative Care; (16.5) Activity; (16.6) Appointments; (16.7) Follow-up Tests / Imaging; (16.8) Call Doctor If; (16.9) Other Instructions.  Use ONLY information present in the note, image, and report; do not introduce facts that are not supported.  Be concise, accurate, and clinically faithful.  Empty sections and subsections may be omitted."

class MMTrainDataset(Dataset):

    def __init__(self, jsonl_path: Path, pilot: int | None=None):
        self._records: list[dict] = []
        for i, line in enumerate(jsonl_path.open()):
            if pilot is not None and i >= pilot:
                break
            self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        r = self._records[idx]
        img_path = DATA_DIR / 'preprocessed' / 'images' / Path(r['cxr_image_path']).name
        img = Image.open(img_path).convert('RGB')
        dicom_id = Path(r['cxr_image_path']).stem
        return {'image': img, 'note': r['input_note'], 'study_id': str(r['study_id']), 'dicom_id': dicom_id, 'report': r.get('radiology_report', ''), 'target': r['target_bhc']}

class CheXagentCollator:

    def __init__(self, tokenizer, image_processor, max_length: int=4096, mode: str='ft', kg: KGRetriever | None=None, n_distractors: int=2, rng_seed: int=42):
        if mode not in ('ft', 'ft_mm', 'raft'):
            raise ValueError(f"mode must be 'ft', 'ft_mm' or 'raft', got {mode!r}")
        if mode == 'raft' and kg is None:
            raise ValueError('RAFT mode requires a KGRetriever instance.')
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.mode = mode
        self.use_image = mode in ('ft_mm', 'raft')
        self.use_kg = mode == 'raft'
        self.kg = kg
        self.n_distractors = n_distractors
        self._rng = random.Random(rng_seed)

    def _text_distractors(self, note: str) -> list[str]:
        if self.kg is None:
            return []
        distractors = self.kg.text_distractors(note, n=self.n_distractors)
        return [f"- {d['head']}  --{d['relation']}-->  {d['tail']}  [DISTRACTOR]" for d in distractors]

    def _kg_text_for(self, note: str, dicom_id: str) -> str:
        real_block = self.kg.build_kg_block(note_text=note, dicom_id=str(dicom_id), section='hospital_course')
        distractor_lines = self._text_distractors(note)
        if real_block and distractor_lines:
            return real_block + '\n' + '\n'.join(distractor_lines)
        if real_block:
            return real_block
        if distractor_lines:
            return 'KNOWN RELATIONS (general medical knowledge — do NOT introduce patient facts not in the note):\n' + '\n'.join(distractor_lines)
        return ''

    def _user_text(self, note: str, dicom_id: str | None, report: str | None=None, report_char_cap: int=1500) -> str:
        parts = [SYSTEM_PROMPT, '', f'Patient note:\n{note}']
        if report and report.strip():
            rep = report.strip()
            if len(rep) > report_char_cap:
                rep = rep[:report_char_cap].rstrip()
            parts.append(f'Paired radiology report:\n{rep}')
        if self.use_kg:
            kg_text = self._kg_text_for(note, dicom_id or '')
            if kg_text:
                parts.append(kg_text)
        parts.append('Generate the Brief Hospital Course:')
        return '\n\n'.join(parts)

    def _build_prompt(self, note: str, dicom_id: str | None, report: str | None=None, target: str | None=None) -> str:
        user_text = self._user_text(note, dicom_id, report)
        prompt = f' USER: <s>{user_text} ASSISTANT: <s>'
        if target is not None:
            prompt += target + self.tokenizer.eos_token
        return prompt

    def _tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    def _fit_to_max_length(self, full_ids: list[int], prompt_ids: list[int]) -> tuple[list[int], int]:
        TARGET_RESERVE = 128
        HEAD_KEEP = 16
        target_ids = full_ids[len(prompt_ids):]
        if not target_ids:
            return (full_ids[:self.max_length], min(len(prompt_ids), self.max_length))
        target_len = min(len(target_ids), TARGET_RESERVE)
        prompt_budget = self.max_length - target_len
        if len(prompt_ids) <= prompt_budget:
            combined = (prompt_ids + target_ids)[:self.max_length]
            return (combined, len(prompt_ids))
        tail_keep = prompt_budget - HEAD_KEEP
        if tail_keep < 0:
            tail_keep = 0
            head_keep = prompt_budget
        else:
            head_keep = HEAD_KEEP
        truncated_prompt = prompt_ids[:head_keep] + prompt_ids[len(prompt_ids) - tail_keep:]
        combined = (truncated_prompt + target_ids)[:self.max_length]
        return (combined, len(truncated_prompt))

    def __call__(self, batch: list[dict]):
        raw_full_ids = [self._tokenize(self._build_prompt(b['note'], b.get('dicom_id'), b.get('report', ''), b['target'])) for b in batch]
        raw_prompt_ids = [self._tokenize(self._build_prompt(b['note'], b.get('dicom_id'), b.get('report', ''), None)) for b in batch]
        full_ids_list, plen_list = ([], [])
        for fids, pids in zip(raw_full_ids, raw_prompt_ids):
            fids_fit, plen = self._fit_to_max_length(fids, pids)
            full_ids_list.append(fids_fit)
            plen_list.append(plen)
        max_len = max((len(ids) for ids in full_ids_list))
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, (fids, plen) in enumerate(zip(full_ids_list, plen_list)):
            flen = len(fids)
            input_ids[i, :flen] = torch.tensor(fids, dtype=torch.long)
            attention_mask[i, :flen] = 1
            labels[i, plen:flen] = torch.tensor(fids[plen:flen], dtype=torch.long)
        for i in range(len(batch)):
            if (labels[i] != -100).sum() == 0:
                valid = (input_ids[i] != pad_id).nonzero(as_tuple=True)[0]
                if len(valid) > 0:
                    labels[i, valid[-1].item()] = input_ids[i, valid[-1].item()].item()
        result = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        if self.use_image:
            imgs = [self.image_processor(b['image'], return_tensors='pt')['pixel_values'][0] for b in batch]
            result['pixel_values'] = torch.stack(imgs).unsqueeze(1)
        return result

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=0.0002)
    ap.add_argument('--max-length', type=int, default=4096)
    ap.add_argument('--model', type=str, default=DEFAULT_MODEL)
    ap.add_argument('--mode', choices=['ft', 'ft_mm', 'raft'], default='ft', help='ft    = note + report (no image, no KG); ft_mm = + CXR image, NO KG, NO distractors (image-matched baseline for raft); raft  = + CXR image + KG block + distractors')
    ap.add_argument('--distractors', type=int, default=2)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--auto-resume', action='store_true')
    ap.add_argument('--fsdp', action='store_true')
    ap.add_argument('--eval-steps', type=int, default=None)
    ap.add_argument('--max-steps', type=int, default=None)
    ap.add_argument('--logging-steps', type=int, default=50)
    args = ap.parse_args()
    if args.fsdp and args.grad_accum > 1:
        print(f'[fsdp] WARNING: overriding grad_accum {args.grad_accum}→1 (FSDP full_shard + no_sync OOMs, cf. feedback_fsdp_oom.md)')
        args.grad_accum = 1
    if not torch.cuda.is_available():
        sys.exit('ERROR: CUDA not available. Use sbatch to submit to a GPU node.')
    spec = get_model(args.model)
    print(f'[model] {spec.short_name}  ({spec.kind})  hf_id={spec.hf_id}')
    _ui = args.mode in ('ft_mm', 'raft')
    _uk = args.mode == 'raft'
    print(f'[mode]  {args.mode}   image={_ui}  kg={_uk}' + (f'  (distractors={args.distractors})' if _uk else ''))
    run_base = spec.run_dir / args.mode / 'fsdp' if args.fsdp else spec.run_dir / args.mode
    if args.auto_resume and args.resume is None:
        if run_base.exists():
            for ts_dir in sorted(run_base.iterdir(), reverse=True):
                if not ts_dir.is_dir():
                    continue
                ckpts = sorted(ts_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
                if ckpts:
                    args.resume = str(ckpts[-1])
                    print(f'[auto-resume] found checkpoint: {args.resume}')
                    break
        if args.resume is None:
            print('[auto-resume] no checkpoint found — starting fresh.')
    if args.resume:
        out_dir = Path(args.resume).parent
        ts = out_dir.name
        state_file = Path(args.resume) / 'trainer_state.json'
        if state_file.exists():
            state = json.loads(state_file.read_text())
            print(f"[resume] epoch {state.get('epoch', '?'):.2f}  global_step {state.get('global_step', '?')}")
        print(f'[resume] output dir → {out_dir}')
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                ts_tensor = torch.tensor([ord(c) for c in ts], dtype=torch.long, device=torch.cuda.current_device())
            else:
                ts_tensor = torch.zeros(15, dtype=torch.long, device=torch.cuda.current_device())
            torch.distributed.broadcast(ts_tensor, src=0)
            ts = ''.join((chr(c) for c in ts_tensor.tolist()))
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = run_base / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[run] output → {out_dir}')
    if int(os.environ.get('RANK', '0')) == 0:
        log_path = out_dir / 'train.log'
        _log_fh = open(log_path, 'w', buffering=1)
        import sys as _sys
        _tee_out = _sys.stdout
        _tee_err = _sys.stderr

        class _Tee:

            def __init__(self, *s):
                self._s = s

            def write(self, d):
                [s.write(d) for s in self._s]

            def flush(self):
                [s.flush() for s in self._s]
        _sys.stdout = _Tee(_tee_out, _log_fh)
        _sys.stderr = _Tee(_tee_err, _log_fh)
        print(f'[log] saving output to {log_path}')
    print(f'[gpu] {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')
    _meta_load = os.environ.get('VERA_CPU_RAM_EFFICIENT', '0') == '1'
    if args.fsdp:
        os.environ['ACCELERATE_USE_FSDP'] = 'true'
        if _meta_load:
            os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING'] = 'true'
        os.environ['FSDP_TRANSFORMER_CLS_TO_WRAP'] = 'MistralDecoderLayer'
        if not torch.distributed.is_initialized():
            import datetime as _dt
            torch.distributed.init_process_group(backend='nccl', timeout=_dt.timedelta(minutes=60))
        _local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(_local_rank)
    _patch_no_mmap_safe_open()
    print(f'[load] processor for {spec.hf_id} (trust_remote_code) ...')
    processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation=os.environ.get('VERA_ATTN', 'eager'))
    print(f"[load] model in bf16 ({('FSDP' if args.fsdp else 'single-GPU')}) ...")
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **load_kwargs)
    if not args.fsdp:
        model = model.to('cuda:0')
    model.config.use_cache = False
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.use_cache = False
    import inspect as _inspect
    _CAC = type(model)
    if not getattr(_CAC, '_kwargs_filtered_patched', False):
        _orig_cls_fwd = _CAC.forward
        _accepted_params = set(_inspect.signature(_orig_cls_fwd).parameters) - {'self'}
        from torch.nn import CrossEntropyLoss as _CELoss

        def _cls_fwd_filtered(self, *args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k in _accepted_params}
            if kwargs.get('pixel_values') is None and kwargs.get('labels') is not None:
                _labels = kwargs.pop('labels')
                out = _orig_cls_fwd(self, *args, **kwargs)
                logits = out.logits if hasattr(out, 'logits') else out[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = _labels[..., 1:].contiguous().to(logits.device)
                loss = _CELoss(reduction='mean')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                return (loss, logits)
            return _orig_cls_fwd(self, *args, **kwargs)
        _CAC.forward = _cls_fwd_filtered
        _CAC._kwargs_filtered_patched = True
        print(f'[fix] Patched {_CAC.__name__}.forward: (1) drop any kwarg not in {sorted(_accepted_params)}, (2) work around the vis_atts UnboundLocalError bug in text-only+labels mode (FT) by computing the shifted causal-LM loss ourselves instead of their buggy path')
    print(f"[lora] r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    if args.fsdp:
        n_cast = 0
        for p in model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
                n_cast += 1
        if n_cast:
            print(f'[dtype] cast {n_cast} float32 param tensors → bfloat16 for FSDP')

    def _split_path(split: str) -> Path:
        cleaned = DATA_DIR / f'multimodal_{split}_cleaned.jsonl'
        return cleaned if cleaned.exists() else DATA_DIR / f'multimodal_{split}_preprocessed.jsonl'
    print(f"[data] train file: {_split_path('train').name}")
    train_ds = MMTrainDataset(_split_path('train'), pilot=args.pilot)
    eval_ds = MMTrainDataset(_split_path('val'), pilot=args.pilot // 5 if args.pilot else 200)
    print(f'[data] train: {len(train_ds):,}  val: {len(eval_ds):,}')
    kg = KGRetriever() if args.mode == 'raft' else None
    collator = CheXagentCollator(tokenizer, processor.image_processor, max_length=args.max_length, mode=args.mode, kg=kg, n_distractors=args.distractors)
    ta_kwargs = dict(output_dir=str(out_dir), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate, lr_scheduler_type='cosine_with_min_lr', lr_scheduler_kwargs={'min_lr_rate': 0.005}, warmup_ratio=0.03, weight_decay=0.0, bf16=True, logging_steps=args.logging_steps, **{'max_steps': args.max_steps} if args.max_steps else {}, eval_strategy='steps' if args.eval_steps else 'no', eval_steps=args.eval_steps, save_strategy='steps', save_steps=200, save_total_limit=10, load_best_model_at_end=args.eval_steps is not None, metric_for_best_model='eval_loss' if args.eval_steps is not None else None, report_to='none', gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, optim='adamw_torch_fused', seed=42, remove_unused_columns=False, dataloader_pin_memory=False, dataloader_num_workers=4, dataloader_prefetch_factor=2, max_grad_norm=0.3)
    if args.fsdp:
        ta_kwargs['fsdp'] = 'full_shard auto_wrap'
        ta_kwargs['fsdp_config'] = {'transformer_layer_cls_to_wrap': ['MistralDecoderLayer'], 'activation_checkpointing': False, **({'fsdp_cpu_ram_efficient_loading': True, 'fsdp_sync_module_states': True} if _meta_load else {})}
    train_args = TrainingArguments(**ta_kwargs)
    from transformers import TrainerCallback as _TCB
    _probe_trainer_holder: list = []

    class _FSDPProbe(_TCB):

        def on_train_begin(self, _a, _s, _c, model=None, **_kw):
            if int(os.environ.get('RANK', '0')) != 0:
                return
            _m = model
            if _m is None and _probe_trainer_holder:
                _m = _probe_trainer_holder[0].model
            if _m is None:
                return
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as _F
                from collections import Counter as _Ctr
                units = [m for m in _m.modules() if isinstance(m, _F)]
                c = _Ctr()
                for u in units:
                    inner = getattr(u, '_fsdp_wrapped_module', None)
                    c[type(inner).__name__ if inner is not None else '?'] += 1
                print(f'[fsdp-probe] total FSDP units={len(units)} by-inner-type={dict(c)}', flush=True)
            except Exception as _e:
                print(f'[fsdp-probe] failed: {_e}', flush=True)
    trainer = Trainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator, callbacks=[_FSDPProbe()])
    _probe_trainer_holder.append(trainer)
    if args.fsdp and os.environ.get('VERA_FSDP_IGNORE_EMB', '1') == '1':
        try:
            ignored = []
            in_emb = model.get_input_embeddings()
            out_emb = model.get_output_embeddings()
            if in_emb is not None:
                ignored.append(in_emb)
            if out_emb is not None and out_emb is not in_emb:
                ignored.append(out_emb)
            _base_model = getattr(model, 'get_base_model', lambda: model)()
            _vision = getattr(_base_model, 'vision_model', None)
            if _vision is not None:
                ignored.append(_vision)
            _qformer = getattr(_base_model, 'qformer', None)
            if _qformer is not None:
                ignored.append(_qformer)
            trainer.accelerator.state.fsdp_plugin.ignored_modules = ignored
            _ld = int(os.environ.get('LOCAL_RANK', 0))
            _dev = torch.device(f'cuda:{_ld}')
            for m in ignored:
                m.to(_dev)
            print(f'[fsdp] excluding {len(ignored)} module(s) (embed/lm_head/vision_model/qformer) from sharding; moved to {_dev}')
        except Exception as e:
            print(f'[fsdp] could not set ignored_modules: {e}')
    if args.fsdp and os.environ.get('VERA_DECODER_ONLY_WRAP', '1') == '1':
        try:
            import functools as _ft
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy as _tap
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer as _MDL
            trainer._fsdp_qlora_plugin_updates = lambda: None
            trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = _ft.partial(_tap, transformer_layer_cls={_MDL})
            print('[fsdp] decoder-only wrap policy (MistralDecoderLayer only)')
        except Exception as e:
            print(f'[fsdp] could not set decoder-only policy: {e}')
    print('[train] starting ...')
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.time() - t0
    print(f'[save] adapters → {out_dir}')
    trainer.save_model(str(out_dir))
    summary = {'model_short_name': spec.short_name, 'model_hf_id': spec.hf_id, 'model_family': spec.family, 'model_kind': spec.kind, 'mode': args.mode, 'quantization': 'bf16', 'n_distractors': args.distractors if args.mode == 'raft' else 0, 'kg_block': args.mode == 'raft', 'cxr_image': args.mode in ('ft_mm', 'raft'), 'lora_config': LORA_CONFIG, 'epochs': args.epochs, 'batch_size': args.batch_size, 'grad_accum': args.grad_accum, 'learning_rate': args.learning_rate, 'max_length': args.max_length, 'pilot': args.pilot, 'train_size': len(train_ds), 'eval_size': len(eval_ds), 'elapsed_seconds': round(elapsed, 1), 'gpu': torch.cuda.get_device_name(0), 'timestamp': ts}
    (out_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'✅ Step 6 (CheXagent-8b) done in {elapsed / 60:.1f} min.')
    return 0
if __name__ == '__main__':
    import traceback as _tb
    _rank = int(os.environ.get('RANK', '0'))
    _log_dir = Path(__file__).parent / 'runs' / 'crash_logs'
    _log_dir.mkdir(parents=True, exist_ok=True)
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as _e:
        _crash_log = _log_dir / f"crash_rank{_rank}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _crash_log.write_text(_tb.format_exc())
        print(f'[rank{_rank}] CRASH → {_crash_log}', flush=True)
        raise
