from __future__ import annotations
import argparse
import glob
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys as _sys
from pathlib import Path as _Path
_cwd = str(_Path.cwd())
if _cwd not in _sys.path:
    _sys.path.insert(0, _cwd)
from models_config import get_model
logger = logging.getLogger('evaluate')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
HERE = Path(__file__).resolve().parent

def _import_local(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
_dataset_mod = _import_local('multimodal_dataset', '05_multimodal_dataset.py')
MultimodalDataset = _dataset_mod.MultimodalDataset
try:
    from rouge_score import rouge_scorer
    _ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
except ImportError:
    _ROUGE = None
    logger.warning('rouge_score not installed; ROUGE will be null.')

def rouge_pair(pred: str, ref: str) -> dict | None:
    if _ROUGE is None:
        return None
    s = _ROUGE.score(ref, pred)
    return {k: v.fmeasure for k, v in s.items()}

def rouge_mean(preds: list[str], refs: list[str]) -> dict | None:
    if _ROUGE is None or not preds:
        return None
    rows = [rouge_pair(p, r) for p, r in zip(preds, refs)]
    return {k: round(sum((r[k] for r in rows)) / len(rows), 4) for k in ('rouge1', 'rouge2', 'rougeL')}
try:
    import bert_score
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False

def bertscore_f1(preds: list[str], refs: list[str]) -> dict | None:
    if not _BERT_AVAILABLE or not preds:
        return None
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        P, R, F1 = bert_score.score(preds, refs, lang='en', rescale_with_baseline=False, device=device)
        return {'bertscore_p': round(float(P.mean()), 4), 'bertscore_r': round(float(R.mean()), 4), 'bertscore_f1': round(float(F1.mean()), 4), 'rescaled': False}
    except Exception as e:
        logger.warning(f'BERTScore failed: {e}')
        return None

def _meteor_available() -> bool:
    try:
        from nltk.translate import meteor_score
        import nltk
        for pkg in ('wordnet', 'omw-1.4'):
            try:
                nltk.data.find(f'corpora/{pkg}')
            except LookupError:
                nltk.download(pkg, quiet=True)
        return True
    except Exception:
        return False

def meteor_mean(preds: list[str], refs: list[str]) -> float | None:
    if not preds or not _meteor_available():
        return None
    try:
        from nltk.translate import meteor_score
        scores = [meteor_score.meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]
        return round(sum(scores) / max(len(scores), 1), 4)
    except Exception as e:
        logger.warning(f'METEOR failed: {e}')
        return None

def per_section_rouge(preds_sectioned: list[dict], refs_sectioned: list[dict]) -> dict:
    if _ROUGE is None or not preds_sectioned:
        return {}
    sections = ['brief_hospital_course', 'radiology_findings', 'radiology_impression']
    aliases = {'brief_hospital_course': ('brief_hospital_course', 'hospital_course')}

    def _pick(rec: dict, sec: str) -> str:
        for k in aliases.get(sec, (sec,)):
            v = rec.get(k)
            if isinstance(v, dict):
                v = ' '.join(v.values())
            if v:
                return v
        return ''
    out: dict[str, dict] = {}
    for sec in sections:
        ps, rs = ([], [])
        for p, r in zip(preds_sectioned, refs_sectioned):
            pv, rv = (_pick(p, sec), _pick(r, sec))
            if pv and rv:
                ps.append(pv)
                rs.append(rv)
        if not ps:
            continue
        rougeL = sum((rouge_pair(p, r)['rougeL'] for p, r in zip(ps, rs))) / len(ps)
        out[sec] = {'rougeL': round(rougeL, 4), 'n_scored': len(ps)}
    return out

def section_completeness(preds_sectioned: list[dict]) -> dict:
    if not preds_sectioned:
        return {}
    n = len(preds_sectioned)
    core = ['brief_hospital_course', 'radiology_findings', 'radiology_impression']
    aliases = {'brief_hospital_course': ('brief_hospital_course', 'hospital_course')}
    out_core = {}
    for sec in core:
        keys = aliases.get(sec, (sec,))
        c = sum((1 for p in preds_sectioned if any((p.get(k) for k in keys))))
        out_core[sec] = round(c / n, 4)
    out_core['overall_mean'] = round(sum((out_core[s] for s in core)) / len(core), 4)
    return out_core
_MED_PATTERNS = [('\\b([a-z][a-z\\-]{3,})\\s+(\\d+(?:\\.\\d+)?)\\s*(mg|mcg|g|mL|units?)\\s*(?:(?:q\\d+h)|(?:po\\s+(?:q?d|bid|tid|qid))|daily|nightly|bid|tid|qid)?', 'with_freq')]

def medication_accuracy(preds: list[str], refs: list[str]) -> float | None:
    if not preds:
        return None
    import re as _re
    pat = _re.compile(_MED_PATTERNS[0][0], _re.I)
    tp = fp = fn = 0
    for p, r in zip(preds, refs):
        ref_meds = {(m.group(1).lower(), m.group(2), m.group(3).lower()) for m in pat.finditer(r)}
        pred_meds = {(m.group(1).lower(), m.group(2), m.group(3).lower()) for m in pat.finditer(p)}
        tp += len(pred_meds & ref_meds)
        fp += len(pred_meds - ref_meds)
        fn += len(ref_meds - pred_meds)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return round(2 * prec * rec / (prec + rec), 4)

def bootstrap_ci(values: list[float], n_boot: int=1000, ci: float=0.95, seed: int=42) -> tuple[float, float]:
    import random
    if not values:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * (1 - ci) / 2)]
    hi = means[int(n_boot * (1 + ci) / 2) - 1]
    return (round(lo, 4), round(hi, 4))

def paired_wilcoxon(scores_a: list[float], scores_b: list[float]) -> dict:
    try:
        from scipy.stats import wilcoxon
        if len(scores_a) != len(scores_b) or len(scores_a) < 5:
            return {'statistic': None, 'pvalue': None, 'reason': 'need ≥5 paired scores'}
        stat, p = wilcoxon(scores_a, scores_b)
        return {'statistic': float(stat), 'pvalue': float(p)}
    except ImportError:
        return {'statistic': None, 'pvalue': None, 'reason': 'scipy not installed'}

def gpu_peak_mem_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    except Exception:
        return None

def cost_per_summary(elapsed_seconds: float, n_summaries: int, gpu_hourly_rate_usd: float=0.5) -> float | None:
    if n_summaries <= 0:
        return None
    hours = elapsed_seconds / 3600
    return round(hours * gpu_hourly_rate_usd / n_summaries, 5)
_UMLS_NLP = None

def _ensure_umls_nlp():
    global _UMLS_NLP
    if _UMLS_NLP is not None:
        return _UMLS_NLP
    try:
        import spacy
        try:
            import scispacy
            from scispacy.linking import EntityLinker
        except ImportError:
            logger.warning('scispaCy not installed; Multimodal MEDCON disabled.')
            _UMLS_NLP = 'missing'
            return _UMLS_NLP
        nlp = spacy.load('en_core_sci_md', disable=['parser'])
        nlp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'umls', 'max_entities_per_mention': 1, 'threshold': 0.85})
        _UMLS_NLP = nlp
        logger.info('MEDCON: scispaCy + UMLS linker loaded')
    except Exception as e:
        logger.warning(f'MEDCON: scispaCy load failed ({e}); metric disabled.')
        _UMLS_NLP = 'missing'
    return _UMLS_NLP

def _text_cuis(text: str) -> set[str]:
    nlp = _ensure_umls_nlp()
    if nlp == 'missing' or not text:
        return set()
    doc = nlp(text)
    cuis: set[str] = set()
    linker = nlp.get_pipe('scispacy_linker')
    for ent in doc.ents:
        for kb_ent_id, _ in ent._.kb_ents[:1]:
            cuis.add(kb_ent_id)
    return cuis
_IMG_EXTRACTOR = None

def _ensure_image_extractor():
    global _IMG_EXTRACTOR
    if _IMG_EXTRACTOR is not None:
        return _IMG_EXTRACTOR
    try:
        import importlib.util, pathlib, sys
        HERE = pathlib.Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('_kg_for_medcon', HERE / '07_kg_retrieval.py')
        _kg = importlib.util.module_from_spec(spec)
        sys.modules['_kg_for_medcon'] = _kg
        spec.loader.exec_module(_kg)
        cig_root = HERE.parent / 'Data' / 'chest-imagenome'
        _IMG_EXTRACTOR = _kg.ImageFindingExtractor(cig_root)
        logger.info(f'Image extractor loaded (base_dir={cig_root})')
    except Exception as e:
        logger.warning(f'Image extractor load failed ({e}); multimodal MEDCON degrades to text-only.')
        _IMG_EXTRACTOR = 'missing'
    return _IMG_EXTRACTOR

def _image_concepts(dicom_id: str) -> set[str]:
    extr = _ensure_image_extractor()
    if extr == 'missing' or not dicom_id:
        return set()
    try:
        findings = extr.extract(dicom_id) or []
        return {f"cig::{f['finding']}" for f in findings if f.get('polarity') == 'present' and f.get('finding')}
    except Exception as e:
        logger.debug(f'image concept extraction failed for {dicom_id}: {e}')
        return set()

def multimodal_medcon_f1(preds: list[str], refs: list[str], dicom_ids: list[str] | None=None) -> dict | None:
    if not preds:
        return None
    nlp = _ensure_umls_nlp()
    if nlp == 'missing':
        return None
    n = len(preds)
    if dicom_ids is None:
        dicom_ids = [''] * n
    text_f1, mm_f1, prec, rec = ([], [], [], [])
    for p, r, did in zip(preds, refs, dicom_ids):
        cui_p = _text_cuis(p)
        cui_r = _text_cuis(r)
        img_p = _image_concepts(did) if did else set()
        img_r = _image_concepts(did) if did else set()
        concepts_p = cui_p | img_p
        concepts_r = cui_r | img_r
        inter = concepts_p & concepts_r
        denom_mm = len(concepts_p) + len(concepts_r)
        denom_tx = len(cui_p) + len(cui_r)
        mm = 2 * len(inter) / denom_mm if denom_mm > 0 else 0.0
        tx = 2 * len(cui_p & cui_r) / denom_tx if denom_tx > 0 else 0.0
        pr = len(inter) / len(concepts_p) if concepts_p else 0.0
        re = len(inter) / len(concepts_r) if concepts_r else 0.0
        mm_f1.append(mm)
        text_f1.append(tx)
        prec.append(pr)
        rec.append(re)
    return {'multimodal_medcon_f1': round(sum(mm_f1) / n, 4), 'text_only_medcon_f1': round(sum(text_f1) / n, 4), 'medcon_precision': round(sum(prec) / n, 4), 'medcon_recall': round(sum(rec) / n, 4), 'n': n}
CHEXBERT_LABELS = ['no_finding', 'enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion', 'pleural_other', 'fracture', 'support_devices']
_CHEXBERT_MODEL = None
_CHEXBERT_HEAD_ORDER = ['enlarged_cardiomediastinum', 'cardiomegaly', 'lung_opacity', 'lung_lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural_effusion', 'pleural_other', 'fracture', 'support_devices', 'no_finding']

def _ensure_chexbert():
    global _CHEXBERT_MODEL
    if _CHEXBERT_MODEL is not None:
        return _CHEXBERT_MODEL
    try:
        import torch
        import torch.nn as nn
        from transformers import BertModel, BertTokenizer
        from huggingface_hub import hf_hub_download

        class _CheXbertLabeler(nn.Module):

            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                self.dropout = nn.Dropout(0.1)
                hidden_size = self.bert.pooler.dense.in_features
                self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
                self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            def forward(self, source_padded, attention_mask):
                final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
                cls_hidden = self.dropout(final_hidden[:, 0, :].squeeze(dim=1))
                return [head(cls_hidden) for head in self.linear_heads]
        ckpt_path = hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', filename='chexbert.pth')
        state = torch.load(ckpt_path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
        model = _CheXbertLabeler()
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(f'CheXbert state_dict partial load: {len(missing)} missing, {len(unexpected)} unexpected keys (missing sample: {missing[:5]}, unexpected sample: {unexpected[:5]})')
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        _CHEXBERT_MODEL = {'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'), 'model': model, 'device': device}
        logger.info('CheXbert loaded: real chexbert.pth checkpoint (StanfordAIMI/RRG_scorers)')
    except Exception as e:
        logger.warning(f'CheXbert unavailable ({e}); using keyword-heuristic fallback.', exc_info=True)
        _CHEXBERT_MODEL = 'fallback'
    return _CHEXBERT_MODEL

def _chexbert_labels(text: str) -> set[str]:
    if not text:
        return set()
    m = _ensure_chexbert()
    if m == 'fallback':
        t = text.lower()
        keywords = {'cardiomegaly': ['cardiomegaly', 'enlarged heart', 'enlarged cardiac'], 'pleural_effusion': ['pleural effusion', 'effusion'], 'consolidation': ['consolidation', 'consolidat'], 'pneumothorax': ['pneumothorax'], 'edema': ['pulmonary edema', 'edema'], 'atelectasis': ['atelectasis', 'atelectatic'], 'lung_opacity': ['opacity', 'opacities', 'opacification'], 'pneumonia': ['pneumonia', 'pneumonic'], 'lung_lesion': ['lung lesion', 'nodule', 'mass'], 'fracture': ['fracture', 'rib fracture'], 'support_devices': ['pacemaker', 'picc', 'endotracheal', 'chest tube', 'ng tube', 'catheter'], 'no_finding': ['no acute cardiopulmonary', 'lungs are clear', 'no focal opacity']}
        return {label for label, kws in keywords.items() if any((k in t for k in kws))}
    import torch
    tok = m['tokenizer'](text[:512], return_tensors='pt', truncation=True, padding=True).to(m['device'])
    with torch.inference_mode():
        out = m['model'](tok['input_ids'], tok['attention_mask'])
    return {name for name, head_logits in zip(_CHEXBERT_HEAD_ORDER, out) if head_logits.squeeze(0).argmax().item() == 1}

def entity_preservation(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    nlp = _ensure_umls_nlp()
    if nlp == 'missing':
        return None
    kept, missed = (0, 0)
    for p, r in zip(preds, refs):
        cui_p = _text_cuis(p)
        cui_r = _text_cuis(r)
        kept += len(cui_p & cui_r)
        missed += len(cui_r - cui_p)
    total = kept + missed
    return {'entity_preservation_recall': round(kept / total, 4) if total else 0.0, 'n_ref_entities': total, 'n_kept': kept}

def section_dependency_score(preds_sectioned: list[dict]) -> dict | None:
    if not preds_sectioned:
        return None
    nlp = _ensure_umls_nlp()
    if nlp == 'missing':
        return None
    pairs = [('discharge_medications', 'discharge_diagnosis'), ('radiology_findings', 'assessment'), ('hospital_course', 'discharge_medications')]
    per_pair = {}
    for a, b in pairs:
        scores = []
        for rec in preds_sectioned:
            ta = rec.get(a) or ''
            tb = rec.get(b) or ''
            if not ta or not tb:
                continue
            ca = _text_cuis(ta)
            cb = _text_cuis(tb)
            union = ca | cb
            if not union:
                continue
            scores.append(len(ca & cb) / len(union))
        per_pair[f'{a}__vs__{b}'] = round(sum(scores) / len(scores), 4) if scores else None
    return {'section_dependency_pairs': per_pair, 'section_dependency_mean': round(sum((v for v in per_pair.values() if v is not None)) / max(sum((1 for v in per_pair.values() if v is not None)), 1), 4)}
_ALIGN_MODEL = None

def _ensure_alignscore():
    global _ALIGN_MODEL
    if _ALIGN_MODEL is not None:
        return _ALIGN_MODEL
    import os
    ckpt = os.environ.get('ALIGNSCORE_CKPT', str(Path(__file__).resolve().parent / 'AlignScore-large.ckpt'))
    if not Path(ckpt).exists():
        logger.warning(f'AlignScore checkpoint not found at {ckpt}; using MedNLI proxy. Set ALIGNSCORE_CKPT env var to override.')
        _ALIGN_MODEL = 'proxy'
        return _ALIGN_MODEL
    try:
        from alignscore import AlignScore
        import torch
        _ALIGN_MODEL = AlignScore(model='roberta-large', batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu', ckpt_path=ckpt, evaluation_mode='nli_sp')
        logger.info(f'AlignScore loaded (real checkpoint: {ckpt})')
    except Exception as e:
        logger.warning(f'AlignScore real model unavailable ({e}); using MedNLI proxy.')
        _ALIGN_MODEL = 'proxy'
    return _ALIGN_MODEL

def alignscore_mean(preds: list[str], refs: list[str]) -> float | None:
    if not preds:
        return None
    m = _ensure_alignscore()
    if m == 'proxy':
        try:
            from transformers import pipeline
            clf = pipeline('text-classification', model='pritamdeka/PubMedBERT-MNLI-MedNLI', top_k=None, device_map='auto')
            scores = []
            for p, r in zip(preds, refs):
                out = clf(f'{r[:400]}[SEP]{p[:400]}', truncation=True)
                for row in out[0]:
                    if row['label'].lower() == 'entailment':
                        scores.append(row['score'])
                        break
            return round(sum(scores) / len(scores), 4) if scores else None
        except Exception as e:
            logger.warning(f'AlignScore proxy failed: {e}')
            return None
    try:
        scores = m.score(contexts=refs, claims=preds)
        return round(sum(scores) / len(scores), 4)
    except Exception as e:
        logger.warning(f'AlignScore failed: {e}')
        return None

def radgraph_f1(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    try:
        from radgraph import F1RadGraph
        f1rg = F1RadGraph(reward_level='partial')
        _, per_example, _, _ = f1rg(refs=refs, hyps=preds)
        return {'radgraph_f1_mean': round(sum(per_example) / len(per_example), 4), 'using_real_model': True, 'n': len(preds)}
    except Exception as e:
        logger.warning(f'RadGraph real model unavailable ({e}); using bigram-overlap proxy.')
    RAD_TERMS = {'lung', 'lungs', 'opacity', 'consolidation', 'effusion', 'pleural', 'cardiomegaly', 'cardiac silhouette', 'mediastinum', 'atelectasis', 'edema', 'pneumothorax', 'nodule', 'mass', 'infiltrate', 'fracture', 'pacemaker', 'catheter', 'picc', 'endotracheal', 'chest tube'}

    def bigrams(text):
        toks = [t for t in text.lower().split() if t in RAD_TERMS]
        return set(zip(toks[:-1], toks[1:])) | set(((t,) for t in toks))
    scores = []
    for p, r in zip(preds, refs):
        bp, br = (bigrams(p), bigrams(r))
        if not (bp or br):
            continue
        inter = len(bp & br)
        denom = len(bp) + len(br)
        scores.append(2 * inter / denom if denom else 0.0)
    return {'radgraph_f1_mean': round(sum(scores) / max(len(scores), 1), 4), 'using_real_model': False, 'n': len(scores)}
_BLEURT_SCORER = None

def _ensure_bleurt():
    global _BLEURT_SCORER
    if _BLEURT_SCORER is not None:
        return _BLEURT_SCORER
    try:
        from bleurt import score
        _BLEURT_SCORER = score.BleurtScorer('BLEURT-20')
        logger.info('BLEURT-20 loaded')
    except Exception as e:
        logger.warning(f'BLEURT unavailable ({e}); metric disabled.')
        _BLEURT_SCORER = 'missing'
    return _BLEURT_SCORER

def bleurt_mean(preds: list[str], refs: list[str]) -> float | None:
    if not preds:
        return None
    m = _ensure_bleurt()
    if m == 'missing':
        return None
    try:
        scores = m.score(references=refs, candidates=preds)
        return round(sum(scores) / len(scores), 4)
    except Exception as e:
        logger.warning(f'BLEURT scoring failed: {e}')
        return None

def per_section_bertscore(preds_sectioned: list[dict], refs_sectioned: list[dict]) -> dict:
    if not _BERT_AVAILABLE or not preds_sectioned:
        return {}
    sections = ['brief_hospital_course', 'radiology_findings', 'radiology_impression']
    aliases = {'brief_hospital_course': ('brief_hospital_course', 'hospital_course')}

    def _pick(rec: dict, sec: str) -> str:
        for k in aliases.get(sec, (sec,)):
            v = rec.get(k)
            if v:
                return v
        return ''
    out = {}
    for sec in sections:
        ps, rs = ([], [])
        for p, r in zip(preds_sectioned, refs_sectioned):
            pv, rv = (_pick(p, sec), _pick(r, sec))
            if pv and rv:
                ps.append(pv)
                rs.append(rv)
        if not ps:
            continue
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _, _, F1 = bert_score.score(ps, rs, lang='en', rescale_with_baseline=False, device=device, verbose=False)
            out[sec] = {'bertscore_f1': round(float(F1.mean()), 4), 'n_scored': len(ps)}
        except Exception as e:
            logger.debug(f'per-section bertscore failed for {sec}: {e}')
    return out

def length_stats(preds: list[str], refs: list[str]) -> dict:
    if not preds:
        return {}
    import numpy as np
    pred_lens = np.array([len(p.split()) for p in preds])
    ref_lens = np.array([len(r.split()) for r in refs])
    return {'pred_len_mean': int(pred_lens.mean()), 'pred_len_median': int(np.median(pred_lens)), 'ref_len_mean': int(ref_lens.mean()), 'ref_len_median': int(np.median(ref_lens)), 'compression_ratio_mean': round(float(pred_lens.mean() / max(ref_lens.mean(), 1)), 3)}

def abstractiveness(preds: list[str], sources: list[str]) -> dict:
    if not preds:
        return {}

    def _ngrams(text: str, n: int) -> set:
        toks = text.lower().split()
        return set((tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))) if len(toks) >= n else set()
    novelty_1, novelty_2, novelty_3 = ([], [], [])
    for p, s in zip(preds, sources):
        for n, out in ((1, novelty_1), (2, novelty_2), (3, novelty_3)):
            p_ng = _ngrams(p, n)
            s_ng = _ngrams(s, n)
            if p_ng:
                out.append(1.0 - len(p_ng & s_ng) / len(p_ng))
    return {'novelty_1gram': round(sum(novelty_1) / max(len(novelty_1), 1), 4), 'novelty_2gram': round(sum(novelty_2) / max(len(novelty_2), 1), 4), 'novelty_3gram': round(sum(novelty_3) / max(len(novelty_3), 1), 4)}

def readability_stats(preds: list[str]) -> dict | None:
    if not preds:
        return None
    try:
        import textstat
    except ImportError:
        logger.warning('textstat not installed — skipping readability')
        return None
    import numpy as np
    fre, fkg, fog = ([], [], [])
    for p in preds:
        if not p.strip():
            continue
        try:
            fre.append(textstat.flesch_reading_ease(p))
            fkg.append(textstat.flesch_kincaid_grade(p))
            fog.append(textstat.gunning_fog(p))
        except Exception:
            continue
    if not fre:
        return None
    return {'flesch_reading_ease_mean': round(float(np.mean(fre)), 2), 'flesch_reading_ease_median': round(float(np.median(fre)), 2), 'flesch_kincaid_grade_mean': round(float(np.mean(fkg)), 2), 'flesch_kincaid_grade_median': round(float(np.median(fkg)), 2), 'gunning_fog_mean': round(float(np.mean(fog)), 2), 'gunning_fog_median': round(float(np.median(fog)), 2), 'n_scored': len(fre)}
_CLINBERT = None

def _ensure_clinicalbert():
    global _CLINBERT
    if _CLINBERT is not None:
        return _CLINBERT
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        mdl = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mdl.to(device)
        _CLINBERT = (tok, mdl, device)
    except Exception as e:
        logger.warning(f'Bio_ClinicalBERT load failed: {e}')
        _CLINBERT = 'missing'
    return _CLINBERT

def clinicalbert_similarity(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    obj = _ensure_clinicalbert()
    if obj == 'missing':
        return None
    import torch, numpy as np
    tok, mdl, device = obj

    def _embed(texts: list[str]) -> 'torch.Tensor':
        with torch.no_grad():
            enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            out = mdl(**enc).last_hidden_state
            mask = enc.attention_mask.unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(pooled, dim=-1)
    sims = []
    B = 8
    for i in range(0, len(preds), B):
        pe = _embed(preds[i:i + B])
        re = _embed(refs[i:i + B])
        sims.extend((pe * re).sum(-1).cpu().tolist())
    return {'clinicalbert_cosine_mean': round(float(np.mean(sims)), 4), 'clinicalbert_cosine_median': round(float(np.median(sims)), 4), 'n_scored': len(sims)}
_TUI_BUCKETS = {'disease': {'T047', 'T048', 'T191', 'T019', 'T037', 'T046', 'T033', 'T184', 'T190'}, 'drug': {'T121', 'T125', 'T129', 'T195', 'T131', 'T197', 'T200', 'T109', 'T116'}, 'procedure': {'T060', 'T061', 'T059', 'T063', 'T058'}}

def _text_cuis_by_bucket(text: str) -> dict[str, set[str]]:
    nlp = _ensure_umls_nlp()
    empty = {b: set() for b in _TUI_BUCKETS}
    if nlp == 'missing' or not text:
        return empty
    doc = nlp(text)
    linker = nlp.get_pipe('scispacy_linker')
    out = {b: set() for b in _TUI_BUCKETS}
    for ent in doc.ents:
        for cui, _ in ent._.kb_ents[:1]:
            try:
                tuis = set(linker.kb.cui_to_entity[cui].types)
            except Exception:
                continue
            for b, keep in _TUI_BUCKETS.items():
                if tuis & keep:
                    out[b].add(cui)
    return out

def cui_f1_by_semtype(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    nlp = _ensure_umls_nlp()
    if nlp == 'missing':
        return None
    tp = {b: 0 for b in _TUI_BUCKETS}
    fp = {b: 0 for b in _TUI_BUCKETS}
    fn = {b: 0 for b in _TUI_BUCKETS}
    for p, r in zip(preds, refs):
        pb = _text_cuis_by_bucket(p)
        rb = _text_cuis_by_bucket(r)
        for b in _TUI_BUCKETS:
            tp[b] += len(pb[b] & rb[b])
            fp[b] += len(pb[b] - rb[b])
            fn[b] += len(rb[b] - pb[b])
    out = {}
    for b in _TUI_BUCKETS:
        denom = 2 * tp[b] + fp[b] + fn[b]
        f1 = 2 * tp[b] / denom if denom > 0 else 0.0
        out[f'{b}_f1'] = round(f1, 4)
        out[f'{b}_tp'] = tp[b]
        out[f'{b}_fp'] = fp[b]
        out[f'{b}_fn'] = fn[b]
    return out

def bertscore_prf(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    try:
        from bert_score import score as bertscore
        P, R, F = bertscore(preds, refs, lang='en', model_type='distilbert-base-uncased', batch_size=32, rescale_with_baseline=True, verbose=False)
        return {'bertscore_p': round(float(P.mean()), 4), 'bertscore_r': round(float(R.mean()), 4), 'bertscore_f1': round(float(F.mean()), 4)}
    except Exception as e:
        logger.warning(f'BERTScore P/R/F1 failed: {e}')
        return None

def chexbert_f1(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    _ensure_chexbert()
    tp = {L: 0 for L in CHEXBERT_LABELS}
    fp = {L: 0 for L in CHEXBERT_LABELS}
    fn = {L: 0 for L in CHEXBERT_LABELS}
    per_record_f1 = []
    for p, r in zip(preds, refs):
        lp = _chexbert_labels(p)
        lr = _chexbert_labels(r)
        for L in CHEXBERT_LABELS:
            in_p = L in lp
            in_r = L in lr
            if in_p and in_r:
                tp[L] += 1
            elif in_p:
                fp[L] += 1
            elif in_r:
                fn[L] += 1
        inter = lp & lr
        denom = len(lp) + len(lr)
        per_record_f1.append(2 * len(inter) / denom if denom > 0 else 0.0)
    macro = 0.0
    for L in CHEXBERT_LABELS:
        denom = 2 * tp[L] + fp[L] + fn[L]
        macro += 2 * tp[L] / denom if denom > 0 else 0.0
    macro /= len(CHEXBERT_LABELS)
    return {'chexbert_f1_macro': round(macro, 4), 'chexbert_f1_per_record_mean': round(sum(per_record_f1) / len(preds), 4), 'chexbert_using_real_model': _CHEXBERT_MODEL != 'fallback', 'n': len(preds)}

def aggregate_verdicts(verdict_path: Path) -> dict:
    import re as _re
    IMG_KWS = _re.compile('\\b(cxr|chest x-?ray|lung|lungs|pulmonary|pleural|effusion|consolidation|infiltrate|opacity|edema|cardiomegaly|mediastin|aortic|hilar|pacemaker|picc|catheter|endotracheal|nasogastric|pneumothorax|atelectasis|radiograph|imaging|nodule|mass|fracture)\\b', _re.IGNORECASE)

    def _label(v: dict) -> str:
        text_nli = v.get('text_nli', 'neutral')
        visual = v.get('visual', 'neutral')
        kg_flag = bool(v.get('kg_flag', False))
        n_pro = int(v.get('n_pro', 0) or 0)
        n_def = int(v.get('n_defeater', 0) or 0)
        if text_nli == 'contradiction' or visual == 'contradiction' or kg_flag or (n_def > 0 and n_def > n_pro):
            return 'REFUTED'
        if (text_nli == 'entailment' or visual == 'entailment') and n_pro >= n_def:
            return 'SUPPORTED'
        return 'NEI'
    n_claims = n_supp = n_ref = n_nei = 0
    n_text_ent = n_text_con = 0
    n_visual_ent = n_visual_con = 0
    n_kg_flag = 0
    n_img_claims = n_img_supp = n_img_ref = 0
    for line in verdict_path.open():
        row = json.loads(line)
        for v in row['verdicts']:
            n_claims += 1
            if v.get('text_nli') == 'entailment':
                n_text_ent += 1
            if v.get('text_nli') == 'contradiction':
                n_text_con += 1
            if v.get('visual') == 'entailment':
                n_visual_ent += 1
            if v.get('visual') == 'contradiction':
                n_visual_con += 1
            if v.get('kg_flag'):
                n_kg_flag += 1
            if IMG_KWS.search(v.get('claim', '')):
                n_img_claims += 1
                lbl = _label(v)
                if lbl == 'SUPPORTED':
                    n_img_supp += 1
                elif lbl == 'REFUTED':
                    n_img_ref += 1
        n_supp += row['summary'].get('SUPPORTED', 0)
        n_ref += row['summary'].get('REFUTED', 0)
        n_nei += row['summary'].get('NEI', 0)
    total = max(n_claims, 1)
    imgtot = max(n_img_claims, 1)
    return {'n_claims': n_claims, 'supported': n_supp, 'refuted': n_ref, 'nei': n_nei, 'supported_rate': round(n_supp / total, 4), 'refuted_rate': round(n_ref / total, 4), 'nei_rate': round(n_nei / total, 4), 'text_nli_entail_rate': round(n_text_ent / total, 4), 'text_nli_contra_rate': round(n_text_con / total, 4), 'visual_entail_rate': round(n_visual_ent / total, 4), 'visual_contra_rate': round(n_visual_con / total, 4), 'kg_contradiction_rate': round(n_kg_flag / total, 4), 'cross_modal': {'n_image_related_claims': n_img_claims, 'cross_modal_supported_rate': round(n_img_supp / imgtot, 4), 'cross_modal_hallucination_rate': round(n_img_ref / imgtot, 4)}}

def _style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.png')
    fig.savefig(out_dir / f'{name}.pdf')
    plt.close(fig)

def plot_training_curves(run_dir: Path, out_dir: Path, tag: str='') -> Path | None:
    state = run_dir / 'trainer_state.json'
    if not state.exists():
        ckpts = sorted(run_dir.glob('checkpoint-*/trainer_state.json'))
        if not ckpts:
            logger.warning(f'No trainer_state.json under {run_dir}')
            return None
        state = ckpts[-1]
    log = json.loads(state.read_text()).get('log_history', [])
    if not log:
        logger.warning(f'Empty log_history in {state}')
        return None
    steps_loss = [(e['step'], e['loss']) for e in log if 'loss' in e and 'eval_loss' not in e]
    steps_eval = [(e['step'], e['eval_loss']) for e in log if 'eval_loss' in e]
    steps_lr = [(e['step'], e['learning_rate']) for e in log if 'learning_rate' in e]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if steps_loss:
        xs, ys = zip(*steps_loss)
        axes[0].plot(xs, ys, label='train loss', color='#1f77b4')
    if steps_eval:
        xs, ys = zip(*steps_eval)
        axes[0].plot(xs, ys, label='eval loss', color='#d62728', marker='o', ms=3)
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('loss')
    axes[0].set_title(f"Training curves{(' — ' + tag if tag else '')}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    if steps_lr:
        xs, ys = zip(*steps_lr)
        axes[1].plot(xs, ys, color='#2ca02c')
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('learning rate')
    axes[1].set_title('Learning-rate schedule')
    axes[1].grid(alpha=0.3)
    name = f"training_curves{('_' + tag if tag else '')}"
    _save(fig, out_dir, name)
    logger.info(f"  plot → {out_dir / (name + '.png')}")
    return out_dir / f'{name}.png'

def plot_faithfulness(verdicts: dict, out_dir: Path, tag: str='') -> Path | None:
    if not verdicts:
        return None
    labels = ['SUPPORTED', 'REFUTED (= hallucination)', 'NEI']
    values = [verdicts['supported_rate'], verdicts['refuted_rate'], verdicts['nei_rate']]
    colors = ['#2ca02c', '#d62728', '#7f7f7f']
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('fraction of claims')
    ax.set_ylim(0, 1)
    ax.set_title(f"Cross-modal Faithfulness{(' — ' + tag if tag else '')}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.2%}', ha='center', va='bottom', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    name = f"faithfulness{('_' + tag if tag else '')}"
    _save(fig, out_dir, name)
    logger.info(f"  plot → {out_dir / (name + '.png')}")
    return out_dir / f'{name}.png'

def plot_tracks_comparison(tracks: list[dict], out_dir: Path) -> Path | None:
    if not tracks or all((t.get('n', 0) == 0 for t in tracks)):
        return None
    metric_names: list[str] = []
    for t in tracks:
        if t.get('rouge_mean'):
            metric_names = list(t['rouge_mean'].keys())
            break
    metric_names = metric_names + ['bertscore_f1']
    track_labels = [t['track'] for t in tracks]
    data: dict[str, list[float]] = {m: [] for m in metric_names}
    for t in tracks:
        for m in metric_names:
            if m == 'bertscore_f1':
                bs = t.get('bertscore_f1')
                if isinstance(bs, dict):
                    data[m].append(bs.get('bertscore_f1', 0.0))
                else:
                    data[m].append(bs or 0.0)
            else:
                rm = t.get('rouge_mean') or {}
                data[m].append(rm.get(m, 0.0))
    _style()
    n_groups = len(metric_names)
    n_tracks = len(track_labels)
    width = 0.8 / max(n_tracks, 1)
    x = list(range(n_groups))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, label in enumerate(track_labels):
        xs = [xi + i * width - 0.4 + width / 2 for xi in x]
        vals = [data[m][i] for m in metric_names]
        ax.bar(xs, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel('score')
    ax.set_title('Reference-based metrics by track')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    _save(fig, out_dir, 'tracks_comparison')
    logger.info(f"  plot → {out_dir / 'tracks_comparison.png'}")
    return out_dir / 'tracks_comparison.png'

def plot_ft_vs_raft(ft: dict, raft: dict, out_dir: Path) -> Path | None:
    ft_f = (ft or {}).get('faithfulness') or {}
    raft_f = (raft or {}).get('faithfulness') or {}
    if not ft_f or not raft_f:
        return None
    metrics = ['supported_rate', 'refuted_rate', 'nei_rate']
    labels = ['SUPPORTED', 'REFUTED (hallucination)', 'NEI']
    ft_vals = [ft_f.get(m, 0.0) for m in metrics]
    raft_vals = [raft_f.get(m, 0.0) for m in metrics]
    _style()
    x = list(range(len(metrics)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([xi - width / 2 for xi in x], ft_vals, width=width, label='FT', color='#1f77b4')
    ax.bar([xi + width / 2 for xi in x], raft_vals, width=width, label='RAFT', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('fraction of claims')
    ax.set_ylim(0, 1)
    ax.set_title('FT vs RAFT — Cross-modal Faithfulness')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    _save(fig, out_dir, 'ft_vs_raft')
    logger.info(f"  plot → {out_dir / 'ft_vs_raft.png'}")
    return out_dir / 'ft_vs_raft.png'

def _evaluate(args):
    spec = get_model(args.model)
    pred_path = Path(args.predictions)
    if not pred_path.exists():
        sys.exit(f'Predictions not found: {pred_path}')
    src_index = {}
    for split in ('test', 'val'):
        try:
            ds = MultimodalDataset(split, keep_in_memory=True)
        except FileNotFoundError:
            continue
        for r in ds._records:
            src_index[r['note_id']] = r
    records = [json.loads(line) for line in pred_path.open()]
    logger.info(f'Loaded {len(records)} predictions')

    def _track(label: str, recs: list[dict]) -> dict:
        if not recs:
            return {'track': label, 'n': 0}
        preds = [r['prediction'] for r in recs]
        refs = [r['target'] for r in recs]
        per_record_rougeL = []
        if _ROUGE is not None:
            for p, r in zip(preds, refs):
                per_record_rougeL.append(rouge_pair(p, r)['rougeL'])

        def _parse_markdown_sections(text: str) -> dict:
            import re
            if not text:
                return {}
            out = {}
            parts = re.split('\\n\\n(##\\s+[^\\n]+)\\n', text)
            if parts[0].strip():
                out['brief_hospital_course'] = parts[0].strip()
            for i in range(1, len(parts), 2):
                header = parts[i].lstrip('# ').strip().lower()
                import re as _re
                key = _re.sub('[^\\w\\s]', '', header).strip().replace(' ', '_')
                key = {'radiology_findings': 'radiology_findings', 'visual_interpretation_of_cxr_model': 'radiology_impression', 'visual_interpretation': 'radiology_impression', 'brief_hospital_course': 'hospital_course'}.get(key, key)
                body = parts[i + 1] if i + 1 < len(parts) else ''
                out[key] = body.strip()
            return out
        preds_sectioned = []
        refs_sectioned = []
        for r in recs:
            pd = r.get('prediction_sections')
            if not pd:
                pd = _parse_markdown_sections(r.get('prediction', ''))
            rd = r.get('target_sections')
            if not rd:
                rd = _parse_markdown_sections(r.get('target', ''))
            preds_sectioned.append(pd)
            refs_sectioned.append(rd)
        dicom_ids = [r.get('dicom_id', '') for r in recs]
        sources = [r.get('source_note') or r.get('input') or '' for r in recs]
        result = {'track': label, 'n': len(recs), 'rouge_mean': rouge_mean(preds, refs), 'rouge_l_95ci': bootstrap_ci(per_record_rougeL) if per_record_rougeL else None, 'bertscore': None if args.no_bertscore else bertscore_f1(preds, refs), 'meteor_mean': meteor_mean(preds, refs), 'bleurt_mean': bleurt_mean(preds, refs), 'medication_accuracy_f1': medication_accuracy(preds, refs), 'section_completeness': section_completeness(preds_sectioned), 'per_section_rougeL': per_section_rouge(preds_sectioned, refs_sectioned), 'per_section_bertscore': None if args.no_bertscore else per_section_bertscore(preds_sectioned, refs_sectioned), 'multimodal_medcon': multimodal_medcon_f1(preds, refs, dicom_ids), 'chexbert_f1': chexbert_f1(preds, refs), 'radgraph_f1': radgraph_f1(preds, refs), 'alignscore_mean': alignscore_mean(preds, refs), 'entity_preservation': entity_preservation(preds, refs), 'clinicalbert_similarity': clinicalbert_similarity(preds, refs), 'cui_f1_by_semtype': cui_f1_by_semtype(preds, refs), 'length_stats': length_stats(preds, refs), 'readability': readability_stats(preds)}
        return result
    full_track = _track('full_cohort', records)
    mm_recs = [r for r in records if r['note_id'] in src_index]
    mm_track = _track('multimodal_paired', mm_recs)
    faithfulness = None
    if args.verdicts:
        v_paths = sorted(glob.glob(args.verdicts))
        if v_paths:
            faithfulness = aggregate_verdicts(Path(v_paths[-1]))
        else:
            logger.warning(f'No verdict files match: {args.verdicts}')
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.outdir) if args.outdir else spec.evaluated_dir
    plots_dir = out_dir / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {'model_short_name': spec.short_name, 'predictions_file': str(pred_path), 'verdicts_file': args.verdicts, 'tracks': [full_track, mm_track], 'faithfulness': faithfulness, 'timestamp': ts}
    out_path = out_dir / f'{pred_path.stem}_metrics_{ts}.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    logger.info(f'Metrics → {out_path}')
    tag = args.tag or ''
    plot_tracks_comparison([full_track, mm_track], plots_dir)
    if faithfulness:
        plot_faithfulness(faithfulness, plots_dir, tag=tag)
    if args.plot_training:
        plot_training_curves(Path(args.plot_training), plots_dir, tag=tag)
    logger.info(f'Plots → {plots_dir}')
    return 0

def _plot_from(args):
    run_dir = Path(args.plot_from)
    if not run_dir.exists():
        sys.exit(f'Run dir not found: {run_dir}')
    plots_dir = run_dir / 'plots'
    tag = args.tag or run_dir.parent.name
    plot_training_curves(run_dir, plots_dir, tag=tag)
    return 0

def _compare(args):
    paths = [Path(p) for p in args.compare]
    if len(paths) != 2:
        sys.exit('--compare requires exactly two metrics JSON paths')
    ft, raft = [json.loads(p.read_text()) for p in paths]
    out_dir = paths[0].parent / 'plots'
    plot_ft_vs_raft(ft, raft, out_dir)
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='medgemma_1.5_4b')
    ap.add_argument('--predictions', default=None, help='<split>_<ts>.jsonl from step 8 (required for full eval)')
    ap.add_argument('--verdicts', default=None, help='<...>_verdicts_*.jsonl glob from step 9')
    ap.add_argument('--no-bertscore', action='store_true', help='Skip BERTScore even if installed (saves GPU time)')
    ap.add_argument('--tag', default='', help="Suffix for plot filenames (e.g. 'ft', 'raft')")
    ap.add_argument('--plot-training', default=None, help='Run dir to plot loss/lr from (in addition to eval)')
    ap.add_argument('--plot-from', default=None, help='Plot training curves only from this run dir; skip metric computation entirely.')
    ap.add_argument('--compare', nargs=2, default=None, help='Two metrics JSONs to plot FT-vs-RAFT comparison')
    ap.add_argument('--outdir', type=str, default=None, help='Explicit output directory for the metrics JSON (e.g. runs/medgemma_27b/evaluated/ft_1000_baseline). Default: spec.evaluated_dir (flat, all configs mixed).')
    args = ap.parse_args()
    if args.plot_from:
        return _plot_from(args)
    if args.compare:
        return _compare(args)
    if not args.predictions:
        sys.exit('--predictions is required unless --plot-from or --compare is used')
    return _evaluate(args)
if __name__ == '__main__':
    raise SystemExit(main())
