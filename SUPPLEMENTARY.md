# Supplementary Material

**Beyond the Radiology Report: What the Chest Radiograph Adds to Discharge Summary Generation Using Vision–Language Models** (ISBI 2027)

This document gives the details that do not fit in the four-page paper: the formal task, the model
architectures, the training hyperparameters, the prompts and the evaluation protocol.

---

## 1. Task

Each record has three sources: a discharge note **N**, the paired radiology report **R** and the
patient's chest radiograph **I**. The output is a three-part discharge summary:

| Part | Symbol | Generated from |
|---|---|---|
| Brief hospital course | **S**<sub>bhc</sub> | **N**, **R** and **I** (fine-tuned model) |
| Radiology findings | **S**<sub>rf</sub> | **R** only (separate pass, fixed prompt) |
| Interpretation of the radiograph | **S**<sub>vi</sub> | **I** only (separate pass, fixed prompt) |

The fine-tuned vision–language model *f* generates the hospital course:

**S**<sub>bhc</sub> = *f*(**N**, **R**, **I**)

**Matched baseline.** For each model, a second configuration *f*<sub>text</sub> is trained on the same
records, with the same settings and number of steps, but with the image removed:

**S**<sub>bhc</sub><sup>text</sup> = *f*<sub>text</sub>(**N**, **R**)

The two configurations differ only in whether the radiograph is supplied, so their difference measures
what the radiograph contributes beyond the radiology report.

---

## 2. Model architectures

All three models follow the same structure: a vision encoder turns the radiograph into image features,
a model-specific connector maps them to visual tokens in the language model's embedding space, and an
autoregressive language-model decoder reads the visual tokens together with the tokenized note, report
and instruction.

| Model | Hugging Face id | Vision encoder; connector |
|---|---|---|
| LLaVA-Med-7B | `microsoft/llava-med-v1.5-mistral-7b` | CLIP ViT-L/14 (336 px); two-layer MLP |
| CheXagent-8b | `StanfordAIMI/CheXagent-8b` | CXR encoder; Q-Former |
| MedGemma-27B | `google/medgemma-27b-it` | SigLIP; MLP projector |

The radiograph is stored at 512 × 512 pixels and resized by each model's processor to its own input size.

---

## 3. Training hyperparameters

| Setting | LLaVA-Med-7B | CheXagent-8b | MedGemma-27B |
|---|---|---|---|
| Adaptation | LoRA | LoRA | LoRA |
| LoRA rank *r* | 16 | 16 | 64 |
| LoRA scaling α | 16 | 16 | 128 |
| LoRA dropout | 0.05 | 0.05 | 0.05 |
| Target modules | q, k, v, o, gate, up, down projections | same | same |
| Learning rate | 2 × 10⁻⁴ | 2 × 10⁻⁴ | 2 × 10⁻⁴ |
| Schedule | cosine with minimum LR (0.005 × peak), warm-up ratio 0.03 | same | same |
| Optimizer | AdamW (fused) | same | same |
| Gradient clipping | 0.3 | 0.3 | 0.3 |
| Epochs | 3 | 3 | 3 |
| Batch size per device | 1 | 1 | 1 |
| Maximum sequence length | 2,048 tokens | 4,096 tokens | 4,096 tokens |
| Precision | bfloat16 | bfloat16 | bfloat16 |
| Parallelism | Fully sharded (FSDP) on AMD MI250X | same | same |
| Seed | 42 | 42 | 42 |
| Training records | 30,741 | 30,741 | 30,741 |

The text-only baseline of each model uses exactly the same settings.

---

## 4. Prompts

### 4.1 Hospital course (fine-tuned model)

```
SYSTEM
You are a clinical writing assistant. Given a patient's discharge note, a chest X-ray image,
and a paired radiology report, write the BRIEF HOSPITAL COURSE (BHC) section. Use ONLY
information present in the note, image, and report. Do NOT introduce facts that are not
supported by the sources.

USER
CLINICAL NOTE:
{note}

PAIRED RADIOLOGY REPORT:
{report}

Write the Brief Hospital Course (BHC):
```

With the radiograph, the image is passed to the model's processor together with this prompt. The
text-only baseline uses the same prompt without the image.

### 4.2 Radiology findings (separate pass, report only)

```
{system prompt above}

Summarize the following chest X-ray report in 2-3 concise sentences. Include only clinically
relevant findings stated in the report.

REPORT:
{report}

SUMMARY:
```

### 4.3 Interpretation of the radiograph (separate pass, image only)

```
{system prompt above}

Describe this chest X-ray in 2-3 concise sentences. Focus on clinically relevant findings only.
Do not speculate beyond what is visible in the image.
```

Each model wraps these prompts in its own chat format (e.g. `[INST] … [/INST]` for LLaVA-Med).

### 4.4 Decoding

| Pass | Maximum new tokens | Decoding |
|---|---|---|
| Hospital course | 512 | greedy |
| Radiology findings | 180 | greedy |
| Interpretation of the radiograph | 180 | greedy |

---

## 5. Data and evaluation protocol

### 5.1 Data
- **MIMIC-IV-Ext-BHC** (discharge notes and reference hospital course), **MIMIC-CXR** (radiology reports)
  and **MIMIC-CXR-JPG** (radiographs), joined into 44,701 note–report–radiograph records.
- Split **at the patient level** into training, validation and test sets (80:10:10), so no patient appears
  in more than one set. Results are reported on a fixed set of **1,000 test records**.
- **Discharge_Me** is used only for external evaluation, without retraining: each record is joined to a
  chest radiology report from the same admission and a chest radiograph of the same patient (1,000 records).

### 5.2 Measures

| Measure | Implementation | Settings |
|---|---|---|
| ROUGE-1, ROUGE-L | `rouge_score` 0.1.2 | Porter stemming |
| METEOR | NLTK 3.9.4 | whitespace tokenization |
| BERTScore P / R / F1 | `bert-score` 0.3.13 | default English model, **no baseline rescaling** |
| RadGraph F1 | `radgraph` 0.1.18 (`F1RadGraph`) | partial reward; mean over records |
| ClinicalBERT similarity | `emilyalsentzer/Bio_ClinicalBERT` | cosine of mean-pooled last-layer embeddings (max. 512 tokens); mean over records |

Each generated summary is compared with the reference summary. All measures are averaged over the
test records.

---

## 6. Data access

The MIMIC datasets and Discharge_Me are available on [PhysioNet](https://physionet.org/) after
credentialing and signing the data use agreement. No patient data is included in this repository.
