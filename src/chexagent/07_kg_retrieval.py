from __future__ import annotations
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
logger = logging.getLogger('kg_retrieval')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
COMMON_ABBREVS: dict[str, str] = {'HTN': 'hypertension', 'DM': 'diabetes mellitus', 'T2DM': 'type 2 diabetes mellitus', 'CAD': 'coronary artery disease', 'CHF': 'congestive heart failure', 'COPD': 'chronic obstructive pulmonary disease', 'MI': 'myocardial infarction', 'AF': 'atrial fibrillation', 'PE': 'pulmonary embolism', 'DVT': 'deep vein thrombosis', 'URI': 'upper respiratory infection', 'UTI': 'urinary tract infection', 'CKD': 'chronic kidney disease', 'GERD': 'gastroesophageal reflux disease', 'OSA': 'obstructive sleep apnea', 'AKI': 'acute kidney injury', 'ARDS': 'acute respiratory distress syndrome'}
_HEADER_AMBIGUOUS = {'PE'}

def abbrev_pattern(abbr: str) -> re.Pattern:
    if abbr in _HEADER_AMBIGUOUS:
        return re.compile(f'\\b{re.escape(abbr)}\\b(?!\\s*:)')
    return re.compile(f'\\b{re.escape(abbr)}\\b')

def expand_abbreviations(text: str) -> str:
    out = text
    for short, long_form in COMMON_ABBREVS.items():
        out = abbrev_pattern(short).sub(long_form, out)
    return out

@dataclass
class Entity:
    text: str
    cui: str = ''
    canonical: str = ''
    types: list[str] = field(default_factory=list)

class TextEntityExtractor:

    def __init__(self):
        self._nlp = None
        try:
            import spacy
            from scispacy.linking import EntityLinker
            self._nlp = spacy.load('en_core_sci_md')
            if 'scispacy_linker' not in self._nlp.pipe_names:
                self._nlp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'umls'})
            logger.info('scispaCy + UMLS linker loaded')
        except Exception as e:
            logger.warning(f'scispaCy/UMLS unavailable ({e}); falling back to heuristic.')
            self._nlp = None

    def extract(self, text: str) -> list[Entity]:
        if self._nlp is not None:
            doc = self._nlp(text)
            entities: list[Entity] = []
            for ent in doc.ents:
                cui = ''
                canonical = ''
                types: list[str] = []
                if hasattr(ent._, 'kb_ents') and ent._.kb_ents:
                    top_cui, _score = ent._.kb_ents[0]
                    cui = top_cui
                    linker = self._nlp.get_pipe('scispacy_linker')
                    canonical = linker.kb.cui_to_entity[cui].canonical_name
                    types = linker.kb.cui_to_entity[cui].types
                entities.append(Entity(text=ent.text, cui=cui, canonical=canonical, types=types))
            return entities
        tokens = set(COMMON_ABBREVS.keys())
        found = [Entity(text=t) for t in tokens if re.search(f'\\b{t}\\b', text)]
        return found

class ImageFindingExtractor:

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self._available = self.base_dir.exists()
        if not self._available:
            logger.warning(f'Chest ImaGenome not found at {self.base_dir}.  Image-side KG returns empty.  Download from PhysioNet and place under Data/chest-imagenome/ to enable.')

    def extract(self, dicom_id: str) -> list[dict]:
        if not self._available:
            return []
        scene_path = self.base_dir / 'silver_dataset' / 'scene_graph' / f'{dicom_id}_SceneGraph.json'
        if not scene_path.exists():
            return []
        try:
            scene = json.loads(scene_path.read_text())
        except Exception as e:
            logger.warning(f'Failed to read {scene_path}: {e}')
            return []
        findings: list[dict] = []
        for block in scene.get('attributes', []):
            anatomy = block.get('bbox_name', '') or block.get('name', '')
            if not anatomy:
                continue
            attr_lists = block.get('attributes', [])
            id_lists = block.get('attributes_ids', [])
            for attrs, ids in zip(attr_lists, id_lists + [[]] * len(attr_lists)):
                for i, finding_str in enumerate(attrs):
                    parts = finding_str.split('|')
                    if len(parts) != 3:
                        continue
                    category, polarity, finding = parts
                    cui = ids[i] if i < len(ids) else ''
                    findings.append({'anatomy': anatomy, 'finding': finding, 'category': category, 'polarity': 'present' if polarity == 'yes' else 'absent', 'cui': cui, 'dicom_id': dicom_id})
        return findings

def is_tier1(fact: dict) -> bool:
    return fact.get('relation') in ('synonym_of', 'abbreviation_of', 'canonical_form', 'umls_canonical')

def is_tier2(fact: dict, note_text: str, image_findings: list[dict]) -> bool:
    head = (fact.get('head') or '').lower()
    tail = (fact.get('tail') or '').lower()
    note_l = note_text.lower()
    img_terms = {f['finding'].lower() for f in image_findings} | {f['anatomy'].lower() for f in image_findings}
    head_present = head in note_l or head in img_terms
    tail_present = tail in note_l or tail in img_terms
    return head_present and tail_present

def grade_relevance(fact: dict, section: str) -> float:
    section_l = section.lower()
    head = (fact.get('head') or '').lower()
    tail = (fact.get('tail') or '').lower()
    rel = (fact.get('relation') or '').lower()
    score = 0.0
    if head and head in section_l:
        score += 0.4
    if tail and tail in section_l:
        score += 0.4
    if is_tier1(fact):
        score += 0.2
    if rel == 'has_finding':
        score += 0.3
        if tail and tail not in ('normal', 'abnormal'):
            score += 0.3
    return min(1.0, score)
SECTIONS_16 = ['patient_and_service', 'chief_complaint', 'hpi', 'past_medical_history', 'medications_admission', 'medications_discharge', 'physical_exam', 'labs_imaging', 'assessment', 'hospital_course', 'discharge_condition', 'radiology_findings', 'radiology_impression', 'procedures_performed', 'discharge_disposition', 'follow_up_and_recommendations']
SUBSECTIONS_RECS = ['consultations', 'safety_flags', 'medication_changes', 'post_operative_care', 'activity', 'appointments', 'followup_tests', 'call_doctor_if', 'other_instructions']
SECTION_DISPLAY_NAMES = {'patient_and_service': 'Patient and Service', 'chief_complaint': 'Chief Complaint / Admission Context', 'hpi': 'History of Present Illness', 'past_medical_history': 'Past Medical / Surgical History', 'medications_admission': 'Medications on Admission', 'medications_discharge': 'Medications (Discharge / Ongoing)', 'physical_exam': 'Physical Examination', 'labs_imaging': 'Investigations / Labs / Imaging', 'assessment': 'Assessment / Impression', 'hospital_course': 'Hospital Course', 'discharge_condition': 'Discharge Condition', 'radiology_findings': 'Radiology Findings', 'radiology_impression': 'Radiology Impression', 'procedures_performed': 'Procedures Performed', 'discharge_disposition': 'Discharge Disposition', 'follow_up_and_recommendations': 'Follow-Up & Recommendations', 'consultations': 'Consultations', 'safety_flags': 'Safety Flags', 'medication_changes': 'Medication Changes', 'post_operative_care': 'Post-operative Care', 'activity': 'Activity', 'appointments': 'Appointments', 'followup_tests': 'Follow-up Tests / Imaging', 'call_doctor_if': 'Call Doctor If', 'other_instructions': 'Other Instructions'}
SECTIONS_15 = SECTIONS_16
SECTIONS_13 = SECTIONS_16
SECTIONS_11 = SECTIONS_16

def retrieve_or_skip(section: str, candidate_facts: list[dict], crag_threshold: float=0.5, max_facts: int=8, max_tier1: int=4) -> list[dict]:
    tier1 = [f for f in candidate_facts if is_tier1(f)]
    others = [f for f in candidate_facts if not is_tier1(f)]
    tier1_scored = sorted(((grade_relevance(f, section), f) for f in tier1), key=lambda x: -x[0])
    tier1_kept = [f for _, f in tier1_scored[:max_tier1]]
    others_scored = sorted(((grade_relevance(f, section), f) for f in others), key=lambda x: -x[0])
    others_kept = [f for s, f in others_scored if s >= crag_threshold]
    combined = tier1_kept + others_kept
    return combined[:max_facts]
UMLS_DISTRACTOR_POOL: list[tuple[str, str, str, str]] = [('actinic keratosis', 'Actinic keratosis', 'Disease', 'C0001168'), ('psoriasis', 'Psoriasis', 'Disease', 'C0033860'), ('ingrown toenail', 'Onychocryptosis', 'Disease', 'C0029863'), ('plantar fasciitis', 'Plantar Fasciitis', 'Disease', 'C0149472'), ('scoliosis', 'Scoliosis', 'Disease', 'C0036439'), ('kyphosis', 'Kyphosis', 'Disease', 'C0022821'), ('myopia', 'Myopia', 'Disease', 'C0027092'), ('hyperopia', 'Hyperopia', 'Disease', 'C0020490'), ('astigmatism', 'Astigmatism', 'Disease', 'C0004128'), ('dental caries', 'Dental Caries', 'Disease', 'C0011334'), ('gingivitis', 'Gingivitis', 'Disease', 'C0017574'), ('temporomandibular disorder', 'Temporomandibular Joint Disorder', 'Disease', 'C0039496'), ('seasonal allergic rhinitis', 'Seasonal Allergic Rhinitis', 'Disease', 'C0085006'), ('vitiligo', 'Vitiligo', 'Disease', 'C0042900'), ('rosacea', 'Rosacea', 'Disease', 'C0035854'), ('seborrheic dermatitis', 'Seborrheic Dermatitis', 'Disease', 'C0011609'), ('strabismus', 'Strabismus', 'Disease', 'C0038379'), ('conjunctivitis', 'Conjunctivitis', 'Disease', 'C0009763'), ('tinnitus', 'Tinnitus', 'Symptom', 'C0040264'), ('vertigo', 'Vertigo', 'Symptom', 'C0042571')]

class KGRetriever:
    _DEFAULT_CIG_DIR = Path(__file__).resolve().parent.parent / 'Data' / 'chest-imagenome'

    def __init__(self, chest_imagenome_dir: Path | None=None, rng_seed: int=42):
        self.text_extractor = TextEntityExtractor()
        self.image_extractor = ImageFindingExtractor(chest_imagenome_dir or self._DEFAULT_CIG_DIR)
        import random
        self._rng = random.Random(rng_seed)

    def text_distractors(self, note_text: str, n: int=2) -> list[dict]:
        note_l = note_text.lower()
        umls_candidates = [(s, c, t, cui) for s, c, t, cui in UMLS_DISTRACTOR_POOL if c.lower() not in note_l and s.lower() not in note_l]
        abbrev_candidates = [(a, COMMON_ABBREVS[a]) for a in COMMON_ABBREVS if not re.search(f'\\b{a}\\b', note_text)]
        n_umls = max(0, int(n * 0.7))
        n_abbr = n - n_umls
        picks_umls = self._rng.sample(umls_candidates, min(n_umls, len(umls_candidates))) if umls_candidates else []
        picks_abbr = self._rng.sample(abbrev_candidates, min(n_abbr, len(abbrev_candidates))) if abbrev_candidates else []
        facts: list[dict] = []
        for s, c, t, cui in picks_umls:
            facts.append({'head': s, 'tail': c, 'relation': 'canonical_form', 'cui': cui, 'semantic_type': t, 'is_distractor': True})
        for a, full in picks_abbr:
            facts.append({'head': a, 'tail': full, 'relation': 'abbreviation_of', 'is_distractor': True})
        return facts

    def candidate_facts(self, note_text: str, image_findings: list[dict], max_umls_entities: int=12) -> list[dict]:
        facts: list[dict] = []
        for abbr, full in COMMON_ABBREVS.items():
            if abbrev_pattern(abbr).search(note_text):
                facts.append({'head': abbr, 'tail': full, 'relation': 'abbreviation_of'})
        entities = self.text_extractor.extract(note_text)
        seen_cuis: set[str] = set()
        for ent in entities:
            if not ent.cui or ent.cui in seen_cuis:
                continue
            seen_cuis.add(ent.cui)
            if len(seen_cuis) > max_umls_entities:
                break
            if ent.canonical and ent.canonical.lower() != ent.text.lower():
                facts.append({'head': ent.text, 'tail': ent.canonical, 'relation': 'canonical_form', 'cui': ent.cui, 'arg_role': 'pro', 'tier': 1})
        for f in image_findings:
            role = 'pro' if f.get('polarity') == 'present' else 'defeater'
            facts.append({'head': f['anatomy'], 'tail': f['finding'], 'relation': 'has_finding' if role == 'pro' else 'absent_finding', 'cui': f.get('cui', ''), 'arg_role': role, 'tier': 2})
        return facts

    def build_kg_block(self, note_text: str, dicom_id: str, section: str, max_facts: int=8, crag_threshold: float=0.5) -> str:
        note_clean = expand_abbreviations(note_text)
        img_findings = self.image_extractor.extract(dicom_id)
        all_facts = self.candidate_facts(note_text, img_findings)
        filtered: list[dict] = []
        for f in all_facts:
            if is_tier1(f) or is_tier2(f, note_clean, img_findings):
                filtered.append(f)
        kept = retrieve_or_skip(section, filtered, crag_threshold=crag_threshold, max_facts=max_facts)
        if not kept:
            return ''
        lines = [f"- {f['head']}  --{f['relation']}-->  {f['tail']}" for f in kept]
        return 'KNOWN RELATIONS (general medical knowledge — do NOT introduce patient facts not in the note):\n' + '\n'.join(lines)

def _smoke():
    print('=== KGRetriever smoke test ===')
    r = KGRetriever()
    note = 'Patient with HTN and CHF, presents with cough and fever. PMH: T2DM. Discharge meds: lisinopril, furosemide.'
    block = r.build_kg_block(note_text=note, dicom_id='synthetic_smoke_no_real_cxr', section='medications_discharge')
    print()
    print(block or '(no KG block — Self-RAG skipped)')
if __name__ == '__main__':
    if '--smoke' in sys.argv:
        _smoke()
    else:
        print(__doc__)
