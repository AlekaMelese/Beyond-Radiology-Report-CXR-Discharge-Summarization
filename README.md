# Beyond the Radiology Report: What the Chest Radiograph Adds to Discharge Summary Generation Using Vision–Language Models

Supplementary material for the ISBI 2027 paper.

The paper asks whether a chest radiograph improves a generated discharge summary when the radiology
report is already an input. Three medical-domain vision–language models (LLaVA-Med-7B, CheXagent-8b and
MedGemma-27B) are fine-tuned with LoRA on the discharge note, the radiology report and the radiograph,
and each is compared with a matched model trained without the radiograph.

## Contents

| File | What it contains |
|---|---|
| [SUPPLEMENTARY.md](SUPPLEMENTARY.md) | Formal task, model architectures, training hyperparameters, prompts, decoding settings and evaluation protocol |
| [LICENSE](LICENSE) | MIT License |

## Data

The MIMIC datasets (MIMIC-IV-Ext-BHC, MIMIC-CXR, MIMIC-CXR-JPG) and Discharge_Me are available on
[PhysioNet](https://physionet.org/) after credentialing and signing the data use agreement.
No patient data is included in this repository.

## Citation

The citation will be added after publication.
