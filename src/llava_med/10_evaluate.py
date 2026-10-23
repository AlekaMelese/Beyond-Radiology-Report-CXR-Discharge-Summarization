from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
logger = logging.getLogger('evaluate')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
try:
    from rouge_score import rouge_scorer
    _ROUGE = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
except ImportError:
    _ROUGE = None
try:
    import bert_score
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False
_CLINBERT = None

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

def radgraph_f1(preds: list[str], refs: list[str]) -> dict | None:
    if not preds:
        return None
    try:
        from radgraph import F1RadGraph
        f1rg = F1RadGraph(reward_level='partial')
        _, per_example, _, _ = f1rg(refs=refs, hyps=preds)
        return {'radgraph_f1': round(sum(per_example) / len(per_example), 4)}
    except Exception as e:
        logger.warning(f'RadGraph failed: {e}')
        return None

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

def main() -> int:
    ap = argparse.ArgumentParser(description='Evaluate generated summaries against the reference summaries.')
    ap.add_argument('predictions', type=str, help='JSONL file written by 08_generate.py (fields: prediction, target)')
    ap.add_argument('--out', type=str, default=None, help='Output JSON file (default: <predictions>_metrics.json)')
    args = ap.parse_args()
    path = Path(args.predictions)
    rows = [json.loads(line) for line in path.open() if line.strip()]
    preds = [r['prediction'] for r in rows]
    refs = [r['target'] for r in rows]
    logger.info(f'Evaluating {len(preds):,} summaries from {path}')
    metrics = {'n': len(preds)}
    rouge = rouge_mean(preds, refs) or {}
    metrics['rouge1'] = rouge.get('rouge1')
    metrics['rougeL'] = rouge.get('rougeL')
    metrics['meteor'] = meteor_mean(preds, refs)
    metrics.update({k: v for k, v in (bertscore_f1(preds, refs) or {}).items() if k != 'rescaled'})
    metrics.update(radgraph_f1(preds, refs) or {})
    metrics.update(clinicalbert_similarity(preds, refs) or {})
    out = Path(args.out) if args.out else path.with_name(path.stem + '_metrics.json')
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0
if __name__ == '__main__':
    sys.exit(main())
