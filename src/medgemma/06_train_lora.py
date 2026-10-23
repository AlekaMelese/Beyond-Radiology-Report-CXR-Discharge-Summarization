from __future__ import annotations
import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
DEFAULT_MODEL = 'medgemma_27b'
LORA_CONFIG = dict(r=64, lora_alpha=128, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
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

class MedGemmaCollator:

    def __init__(self, processor, max_length: int=4096, mode: str='ft', kg: KGRetriever | None=None, n_distractors: int=2, rng_seed: int=42):
        if mode not in ('ft', 'ft_mm', 'raft'):
            raise ValueError(f"mode must be 'ft', 'ft_mm' or 'raft', got {mode!r}")
        if mode == 'raft' and kg is None:
            raise ValueError('RAFT mode requires a KGRetriever instance.')
        self.processor = processor
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
        parts = [f'Patient note:\n{note}']
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

    def _messages(self, image, note: str, dicom_id: str | None, report: str | None=None, target: str | None=None):
        user_content: list[dict] = []
        if self.use_image:
            user_content.append({'type': 'image', 'image': image})
        user_content.append({'type': 'text', 'text': self._user_text(note, dicom_id, report)})
        msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]}, {'role': 'user', 'content': user_content}]
        if target is not None:
            msgs.append({'role': 'assistant', 'content': [{'type': 'text', 'text': target}]})
        return msgs

    def __call__(self, batch: list[dict]):
        full_msgs = [self._messages(b['image'], b['note'], b.get('dicom_id'), report=b.get('report', ''), target=b['target']) for b in batch]
        prompt_msgs = [self._messages(b['image'], b['note'], b.get('dicom_id'), report=b.get('report', ''), target=None) for b in batch]

        def _one_str(out):
            if isinstance(out, str):
                return out
            if isinstance(out, list) and all((isinstance(x, str) for x in out)):
                return ''.join(out)
            raise TypeError(f'apply_chat_template returned {type(out)}: {out!r}')
        full_text = [_one_str(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)) for msgs in full_msgs]
        prompt_text = [_one_str(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)) for msgs in prompt_msgs]
        proc_kwargs = dict(text=full_text, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        if self.use_image:
            proc_kwargs['images'] = [[b['image']] for b in batch]
        encoded = self.processor(**proc_kwargs)
        labels = encoded['input_ids'].clone()
        for i, p_text in enumerate(prompt_text):
            prompt_proc_kwargs = dict(text=[p_text], padding=False, truncation=True, max_length=self.max_length, return_tensors='pt')
            if self.use_image:
                prompt_proc_kwargs['images'] = [[batch[i]['image']]]
            enc_prompt = self.processor(**prompt_proc_kwargs)
            prompt_len = enc_prompt['input_ids'].shape[-1]
            labels[i, :prompt_len] = -100
        labels[encoded['input_ids'] == self.processor.tokenizer.pad_token_id] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        for i in range(labels.shape[0]):
            if (labels[i] != -100).sum() == 0:
                non_pad = (encoded['input_ids'][i] != pad_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    labels[i, non_pad[-1].item()] = encoded['input_ids'][i, non_pad[-1].item()].item()
        encoded['labels'] = labels
        return encoded

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', type=int, default=None, help='If set, use only the first N records (smoke test).')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--learning-rate', type=float, default=0.0002)
    ap.add_argument('--max-length', type=int, default=4096)
    ap.add_argument('--model', type=str, default=DEFAULT_MODEL, help='short_name from models_config.py (e.g. medgemma_1.5_4b, llava_med_v1.5_7b, ...)')
    ap.add_argument('--mode', choices=['ft', 'ft_mm', 'raft'], default='ft', help='ft    = plain SFT, note + report (no image, no KG); ft_mm = + CXR image, still NO KG and NO distractors (image-matched baseline for raft -- without it the ft<->raft contrast moves image + KG + distractors together, cf. plan 9.1.1); raft  = + CXR image + KG block + N distractor facts.')
    ap.add_argument('--distractors', type=int, default=2, help='(raft mode only) number of distractor facts per sample.')
    ap.add_argument('--resume', type=str, default=None, help='Path to a specific checkpoint dir to resume from.')
    ap.add_argument('--auto-resume', action='store_true', help='Automatically find and resume from the latest checkpoint for this model+mode combination.')
    ap.add_argument('--fsdp', action='store_true', help='Use FSDP (true parallel, launch with torchrun). Default off = device_map=auto single-process pipeline.')
    ap.add_argument('--eval-steps', type=int, default=None, help="If set, run evaluation on the val split every N steps (eval_strategy='steps'). Default None = no eval.")
    ap.add_argument('--max-steps', type=int, default=None, help='If set, stop after N optimizer steps (overrides epochs). Use with a small --pilot for a quick loss probe.')
    ap.add_argument('--logging-steps', type=int, default=50, help='Log training loss every N steps (default 50).')
    args = ap.parse_args()
    if args.fsdp and args.grad_accum > 1:
        print(f'[fsdp] WARNING: grad_accum={args.grad_accum} OOMs with FSDP full_shard on 64 GB GCDs: no_sync accumulates full (unsharded) gradient buffers for all ~62 FSDP units simultaneously (~54 GB bf16). Overriding to grad_accum=1.')
        args.grad_accum = 1
    if not torch.cuda.is_available():
        sys.exit('ERROR: CUDA not available. Submit this script to a GPU node via SLURM (sbatch submit_train.sh).')
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
            print(f"[resume] epoch {state.get('epoch', '?'):.2f}  global_step {state.get('global_step', '?')}  best_loss {state.get('best_metric', '?')}")
        print(f'[resume] output dir → {out_dir}')
    else:
        if dist.is_initialized():
            if dist.get_rank() == 0:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                ts_tensor = torch.tensor([ord(c) for c in ts], dtype=torch.long, device=torch.cuda.current_device())
            else:
                ts_tensor = torch.zeros(15, dtype=torch.long, device=torch.cuda.current_device())
            dist.broadcast(ts_tensor, src=0)
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

            def __init__(self, *streams):
                self._s = streams

            def write(self, d):
                for s in self._s:
                    s.write(d)

            def flush(self):
                for s in self._s:
                    s.flush()
        _sys.stdout = _Tee(_tee_out, _log_fh)
        _sys.stderr = _Tee(_tee_err, _log_fh)
        print(f'[log] saving output to {log_path}')
    print(f'[gpu] {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB')
    _meta_load = os.environ.get('VERA_CPU_RAM_EFFICIENT', '0') == '1'
    if args.fsdp:
        os.environ['ACCELERATE_USE_FSDP'] = 'true'
        if _meta_load:
            os.environ['FSDP_CPU_RAM_EFFICIENT_LOADING'] = 'true'
        os.environ['FSDP_TRANSFORMER_CLS_TO_WRAP'] = 'Gemma3DecoderLayer,SiglipEncoderLayer'
        if not torch.distributed.is_initialized():
            import datetime as _dt
            torch.distributed.init_process_group(backend='nccl', timeout=_dt.timedelta(minutes=60))
        _local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(_local_rank)
    print(f'[load] processor for {spec.hf_id} ...')
    processor = AutoProcessor.from_pretrained(spec.hf_id, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation=os.environ.get('VERA_ATTN', 'sdpa'), trust_remote_code=True)
    if not args.fsdp:
        load_kwargs['device_map'] = 'auto'
    print(f"[load] model in bf16 ({('FSDP' if args.fsdp else 'device_map=auto')}) ...")
    model = AutoModelForImageTextToText.from_pretrained(spec.hf_id, **load_kwargs)
    model.config.use_cache = False
    if hasattr(model.config, 'text_config'):
        model.config.text_config.use_cache = False
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.use_cache = False
    for _cfg in [model.config, getattr(model.config, 'text_config', None)]:
        if _cfg is not None and hasattr(_cfg, 'cache_implementation'):
            _cfg.cache_implementation = None
    try:
        from transformers.cache_utils import Cache as _Cache
        if not hasattr(_Cache, 'float'):
            _Cache.float = lambda self: self
        if not hasattr(_Cache, 'to'):
            _Cache.to = lambda self, *a, **kw: self
    except ImportError:
        pass
    if args.fsdp:
        out_emb = model.get_output_embeddings()
        in_emb = model.get_input_embeddings()
        if out_emb is not None and in_emb is not None:
            out_emb.weight = torch.nn.Parameter(out_emb.weight.data.clone())
            model.config.tie_word_embeddings = False
            if hasattr(model.config, 'text_config'):
                model.config.text_config.tie_word_embeddings = False
            print(f"[fsdp] untied lm_head (rank {os.environ.get('RANK', '?')}) for FSDP compatibility")
    print(f"[lora] r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    if args.fsdp:
        for p in model.parameters():
            if p.requires_grad and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)

    def _split_path(split: str) -> Path:
        cleaned = DATA_DIR / f'multimodal_{split}_cleaned.jsonl'
        return cleaned if cleaned.exists() else DATA_DIR / f'multimodal_{split}_preprocessed.jsonl'
    print(f"[data] train file: {_split_path('train').name}")
    train_ds = MMTrainDataset(_split_path('train'), pilot=args.pilot)
    eval_ds = MMTrainDataset(_split_path('val'), pilot=args.pilot // 5 if args.pilot else 200)
    print(f'[data] train: {len(train_ds):,}  val: {len(eval_ds):,}')
    kg = KGRetriever() if args.mode == 'raft' else None
    collator = MedGemmaCollator(processor, max_length=args.max_length, mode=args.mode, kg=kg, n_distractors=args.distractors)
    ta_kwargs = dict(output_dir=str(out_dir), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, learning_rate=args.learning_rate, lr_scheduler_type='cosine_with_min_lr', lr_scheduler_kwargs={'min_lr_rate': 0.005}, warmup_ratio=0.03, weight_decay=0.0, bf16=True, logging_steps=args.logging_steps, **{'max_steps': args.max_steps} if args.max_steps else {}, eval_strategy='steps' if args.eval_steps else 'no', eval_steps=args.eval_steps, save_strategy='steps', save_steps=200, save_total_limit=10, load_best_model_at_end=args.eval_steps is not None, metric_for_best_model='eval_loss' if args.eval_steps is not None else None, report_to='none', gradient_checkpointing=not args.fsdp, gradient_checkpointing_kwargs={'use_reentrant': False} if not args.fsdp else {}, optim='adamw_torch_fused', seed=42, remove_unused_columns=False, dataloader_pin_memory=False, dataloader_num_workers=4, dataloader_prefetch_factor=2, max_grad_norm=0.3)
    if args.fsdp:
        ta_kwargs['fsdp'] = 'full_shard auto_wrap'
        ta_kwargs['fsdp_config'] = {'transformer_layer_cls_to_wrap': ['Gemma3DecoderLayer', 'SiglipEncoderLayer'], 'activation_checkpointing': True, **({'fsdp_cpu_ram_efficient_loading': True, 'fsdp_sync_module_states': True} if _meta_load else {})}
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
                print('[fsdp-probe] model unavailable at on_train_begin', flush=True)
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
            trainer.accelerator.state.fsdp_plugin.ignored_modules = ignored
            _ld = int(os.environ.get('LOCAL_RANK', 0))
            _dev = torch.device(f'cuda:{_ld}')
            for m in ignored:
                m.to(_dev)
            print(f'[fsdp] excluding {len(ignored)} module(s) (embed/lm_head) from sharding; moved to {_dev}')
        except Exception as e:
            print(f'[fsdp] could not set ignored_modules: {e}')
    if args.fsdp and os.environ.get('VERA_DECODER_ONLY_WRAP', '1') == '1':
        try:
            import functools as _ft
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy as _tap
            from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer as _GDL
            trainer._fsdp_qlora_plugin_updates = lambda: None
            trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = _ft.partial(_tap, transformer_layer_cls={_GDL})
            print('[fsdp] decoder-only wrap policy (Gemma3DecoderLayer only)')
        except Exception as e:
            print(f'[fsdp] could not set decoder-only policy: {e}')
    print(f'[train] starting ...')
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume)
    elapsed = time.time() - t0
    print(f'[save] adapters → {out_dir}')
    trainer.save_model(str(out_dir))
    summary = {'model_short_name': spec.short_name, 'model_hf_id': spec.hf_id, 'model_family': spec.family, 'model_kind': spec.kind, 'mode': args.mode, 'quantization': 'bf16', 'n_distractors': args.distractors if args.mode == 'raft' else 0, 'kg_block': args.mode == 'raft', 'cxr_image': args.mode in ('ft_mm', 'raft'), 'lora_config': LORA_CONFIG, 'epochs': args.epochs, 'batch_size': args.batch_size, 'grad_accum': args.grad_accum, 'learning_rate': args.learning_rate, 'max_length': args.max_length, 'pilot': args.pilot, 'train_size': len(train_ds), 'eval_size': len(eval_ds), 'elapsed_seconds': round(elapsed, 1), 'gpu': torch.cuda.get_device_name(0), 'timestamp': ts}
    (out_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'✅ Step 6 done in {elapsed / 60:.1f} min.')
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
