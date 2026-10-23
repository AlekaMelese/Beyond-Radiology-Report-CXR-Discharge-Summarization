from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / 'runs'

@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    hf_id: str
    family: str
    kind: str
    image_input_size: int
    notes: str = ''

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.short_name

    @property
    def generated_dir(self) -> Path:
        return self.run_dir / 'generated'

    @property
    def evaluated_dir(self) -> Path:
        return self.run_dir / 'evaluated'

    def adapters_dir(self, timestamp: str) -> Path:
        return self.run_dir / timestamp
CORE_MODELS: dict[str, ModelSpec] = {'medgemma_1.5_4b': ModelSpec(short_name='medgemma_1.5_4b', hf_id='google/medgemma-1.5-4b-it', family='Gemma', kind='medical', image_input_size=448, notes=''), 'medgemma_27b': ModelSpec(short_name='medgemma_27b', hf_id='google/medgemma-27b-it', family='Gemma', kind='medical', image_input_size=448, notes=''), 'llava_med_v1.5_7b': ModelSpec(short_name='llava_med_v1.5_7b', hf_id='microsoft/llava-med-v1.5-mistral-7b', family='Mistral/LLaVA', kind='medical', image_input_size=336, notes=''), 'chexagent_2_8b': ModelSpec(short_name='chexagent_2_8b', hf_id='StanfordAIMI/CheXagent-2-3b', family='LLaMA + CXR encoder', kind='cxr-specialist', image_input_size=448, notes=''), 'qwen3_vl_8b': ModelSpec(short_name='qwen3_vl_8b', hf_id='Qwen/Qwen3-VL-8B-Instruct', family='Qwen3', kind='general', image_input_size=448, notes=''), 'phi4_multimodal': ModelSpec(short_name='phi4_multimodal', hf_id='microsoft/Phi-4-multimodal-instruct', family='Phi-4', kind='general', image_input_size=448, notes=''), 'paligemma2_3b': ModelSpec(short_name='paligemma2_3b', hf_id='google/paligemma2-3b-pt-448', family='Gemma 2 + SigLIP', kind='general', image_input_size=448, notes='')}
RESERVE_MODELS: dict[str, ModelSpec] = {'internvl2.5_8b': ModelSpec(short_name='internvl2.5_8b', hf_id='OpenGVLab/InternVL2_5-8B', family='InternLM + InternViT', kind='general', image_input_size=448, notes=''), 'maira2': ModelSpec(short_name='maira2', hf_id='microsoft/maira-2', family='LLaMA + image encoder', kind='medical', image_input_size=512, notes='')}
ALL_MODELS: dict[str, ModelSpec] = {**CORE_MODELS, **RESERVE_MODELS}

def get_model(name: str) -> ModelSpec:
    if name not in ALL_MODELS:
        raise KeyError(f"Unknown model '{name}'.  Available: {sorted(ALL_MODELS.keys())}")
    return ALL_MODELS[name]

def list_models(kind: str | None=None) -> list[ModelSpec]:
    return [m for m in ALL_MODELS.values() if kind is None or m.kind == kind]
if __name__ == '__main__':
    print('=== Core models ===')
    for m in CORE_MODELS.values():
        print(f'  {m.short_name:22s}  [{m.kind:14s}]  {m.hf_id}')
    print()
    print('=== Reserve models ===')
    for m in RESERVE_MODELS.values():
        print(f'  {m.short_name:22s}  [{m.kind:14s}]  {m.hf_id}')
    print()
    print(f'Folders under: {RUNS_DIR}')
