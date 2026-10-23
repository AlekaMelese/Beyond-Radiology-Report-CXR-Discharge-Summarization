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
CORE_MODELS: dict[str, ModelSpec] = {'chexagent_8b': ModelSpec(short_name='chexagent_8b', hf_id='StanfordAIMI/CheXagent-8b', family='LLaMA + CXR encoder', kind='medical', image_input_size=448, notes='')}
RESERVE_MODELS: dict[str, ModelSpec] = {}
ALL_MODELS: dict[str, ModelSpec] = {**CORE_MODELS, **RESERVE_MODELS}

def get_model(name: str) -> ModelSpec:
    if name not in ALL_MODELS:
        raise KeyError(f"Unknown model '{name}'.  Available: {sorted(ALL_MODELS.keys())}")
    return ALL_MODELS[name]

def list_models(kind: str | None=None) -> list[ModelSpec]:
    return [m for m in ALL_MODELS.values() if kind is None or m.kind == kind]
if __name__ == '__main__':
    for m in ALL_MODELS.values():
        print(f'  {m.short_name:25s}  [{m.kind:14s}]  {m.hf_id}')
    print(f'\nRuns dir: {RUNS_DIR}')
