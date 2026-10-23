# Beyond the Radiology Report: What the Chest Radiograph Adds to Discharge Summary Generation Using Vision–Language Models

Supplementary material for the ISBI 2027 paper.

The paper asks whether a chest radiograph improves a generated discharge summary when the radiology
report is already an input. Three medical-domain vision–language models (LLaVA-Med-7B, CheXagent-8b and
MedGemma-27B) are fine-tuned with LoRA on the discharge note, the radiology report and the radiograph,
and each is compared with a matched model trained without the radiograph.

This page gives the details that do not fit in the four-page paper: the formal task, the model
architectures, the training hyperparameters, the prompts and the evaluation protocol.

## Contents
1. [Task](#1-task)
2. [Model architectures](#2-model-architectures)
3. [Mathematical formulation](#3-mathematical-formulation)
4. [Training hyperparameters](#4-training-hyperparameters)
5. [Prompts](#5-prompts)
6. [Data and evaluation protocol](#6-data-and-evaluation-protocol)
7. [Data access](#7-data-access)
8. [Repository structure and code](#8-repository-structure-and-code)
9. [Example generated summary](#9-example-generated-summary)

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

## 3. Mathematical formulation

### 3.1 Input encoding
The vision encoder $h$ splits the radiograph $\mathbf{I}$ into patches and produces $T$ patch features of
dimension $D_v$; the connector $g$ maps them to $K$ visual tokens in the decoder's embedding space of
dimension $d$:

$$\mathbf{V} = g\big(h(\mathbf{I})\big) \in \mathbb{R}^{K \times d}.$$

The instruction, the note $\mathbf{N}$ and the report $\mathbf{R}$ are tokenized and embedded as
$\mathbf{E} \in \mathbb{R}^{M \times d}$. The decoder input is the sequence $[\mathbf{V}; \mathbf{E}]$
with the radiograph, and $\mathbf{E}$ alone for the text-only baseline.

### 3.2 Generation
The decoder generates the hospital course $\mathbf{y} = (y_1, \dots, y_L)$ autoregressively:

$$p_\theta(\mathbf{y} \mid \mathbf{N}, \mathbf{R}, \mathbf{I}) = \prod_{t=1}^{L} p_\theta\big(y_t \mid y_{<t}, \mathbf{V}, \mathbf{E}\big),$$

with greedy decoding, $y_t = \arg\max_{w} p_\theta(w \mid y_{<t}, \mathbf{V}, \mathbf{E})$.

### 3.3 Training objective
Each model is fine-tuned by minimizing the negative log-likelihood of the reference hospital course
$\mathbf{y}^\ast$, computed **on the target tokens only** (prompt and padding tokens are masked out of the loss):

$$\mathcal{L}(\theta) = -\sum_{t=1}^{|\mathbf{y}^\ast|} \log p_\theta\big(y^\ast_t \mid y^\ast_{<t}, \mathbf{V}, \mathbf{E}\big).$$

### 3.4 Low-rank adaptation (LoRA)
For each adapted weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ of the decoder
(query, key, value, output and the three feed-forward projections), LoRA learns a low-rank update while
$W_0$ stays frozen:

$$W = W_0 + \frac{\alpha}{r}\, B A, \qquad B \in \mathbb{R}^{d_{\text{out}} \times r},\quad A \in \mathbb{R}^{r \times d_{\text{in}}},\quad r \ll \min(d_{\text{in}}, d_{\text{out}}).$$

Only $A$ and $B$ are trained; $r$ and $\alpha$ for each model are given in Section 4.

### 3.5 Evaluation measures
For a generated summary $\hat{\mathbf{y}}$ and the reference $\mathbf{y}^\ast$:

- **ROUGE-N (F1).** With $G_n(\cdot)$ the multiset of $n$-grams,

$$P_n = \frac{|G_n(\hat{\mathbf{y}}) \cap G_n(\mathbf{y}^\ast)|}{|G_n(\hat{\mathbf{y}})|}, \qquad R_n = \frac{|G_n(\hat{\mathbf{y}}) \cap G_n(\mathbf{y}^\ast)|}{|G_n(\mathbf{y}^\ast)|}, \qquad F_n = \frac{2 P_n R_n}{P_n + R_n}.$$

  ROUGE-L uses the longest common subsequence in place of the $n$-gram overlap.
- **BERTScore.** With contextual token embeddings $\mathbf{e}$,

$$R_{\text{BERT}} = \frac{1}{|\mathbf{y}^\ast|}\sum_{u \in \mathbf{y}^\ast} \max_{v \in \hat{\mathbf{y}}} \cos(\mathbf{e}_u, \mathbf{e}_v), \qquad P_{\text{BERT}} = \frac{1}{|\hat{\mathbf{y}}|}\sum_{v \in \hat{\mathbf{y}}} \max_{u \in \mathbf{y}^\ast} \cos(\mathbf{e}_u, \mathbf{e}_v), \qquad F_{\text{BERT}} = \frac{2 P_{\text{BERT}} R_{\text{BERT}}}{P_{\text{BERT}} + R_{\text{BERT}}}.$$

- **METEOR.** A harmonic mean of unigram precision and recall (recall weighted higher) after exact, stem
  and synonym matching, with a penalty for fragmented matches.
- **RadGraph F1.** The F1 score between the clinical entities and relations that RadGraph extracts from
  $\hat{\mathbf{y}}$ and from $\mathbf{y}^\ast$ (partial-reward setting).
- **ClinicalBERT similarity.** With $\bar{\mathbf{e}}(\cdot)$ the mean-pooled Bio_ClinicalBERT embedding,

$$\text{sim}(\hat{\mathbf{y}}, \mathbf{y}^\ast) = \cos\big(\bar{\mathbf{e}}(\hat{\mathbf{y}}), \bar{\mathbf{e}}(\mathbf{y}^\ast)\big).$$

Every measure is computed per record and averaged over the test records.

---

## 4. Training hyperparameters

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

## 5. Prompts

### 5.1 Hospital course during training

During fine-tuning, the hospital course is learned with a longer system prompt that lists the sections a
discharge summary may contain (shown for LLaVA-Med-7B; the radiograph token is omitted for the text-only
baseline, and the report is cut to its first 1,500 characters):

```
[INST] <image>
You are a clinical summarizer. Generate the Brief Hospital Course (BHC) narrative from the patient's
clinical note, chest X-ray, and paired radiology report.  The narrative should cover, in order: (1)
Patient and Service; (2) Chief Complaint / Admission Context; (3) History of Present Illness; (4)
Past Medical / Surgical History; (5) Medications on Admission; (6) Medications (Discharge /
Ongoing); (7) Physical Examination; (8) Investigations / Labs / Imaging; (9) Assessment /
Impression; (10) Hospital Course; (11) Discharge Condition; (12) Radiology Findings; (13) Radiology
Impression; (14) Procedures Performed; (15) Discharge Disposition; (16) Follow-Up & Recommendations
— composed of conditional subsections rendered ONLY when the input supports them: (16.1)
Consultations; (16.2) Safety Flags (drug interactions, allergies, contraindications); (16.3)
Medication Changes; (16.4) Post-operative Care; (16.5) Activity; (16.6) Appointments; (16.7) Follow-
up Tests / Imaging; (16.8) Call Doctor If; (16.9) Other Instructions.  Use ONLY information present
in the note, image, and report; do not introduce facts that are not supported.  Be concise,
accurate, and clinically faithful.  Empty sections and subsections may be omitted.

Patient note:
{note}

Paired radiology report:
{report}

Generate the Brief Hospital Course: [/INST]
```

### 5.2 Hospital course at generation

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

### 5.3 Radiology findings (separate pass, report only)

```
{system prompt above}

Summarize the following chest X-ray report in 2-3 concise sentences. Include only clinically
relevant findings stated in the report.

REPORT:
{report}

SUMMARY:
```

### 5.4 Interpretation of the radiograph (separate pass, image only)

```
{system prompt above}

Describe this chest X-ray in 2-3 concise sentences. Focus on clinically relevant findings only.
Do not speculate beyond what is visible in the image.
```

Each model wraps these prompts in its own chat format (e.g. `[INST] … [/INST]` for LLaVA-Med).

### 5.5 Decoding

| Pass | Maximum new tokens | Decoding |
|---|---|---|
| Hospital course | 512 | greedy |
| Radiology findings | 180 | greedy |
| Interpretation of the radiograph | 180 | greedy |

---

## 6. Data and evaluation protocol

### 6.1 Data
- **MIMIC-IV-Ext-BHC** (discharge notes and reference hospital course), **MIMIC-CXR** (radiology reports)
  and **MIMIC-CXR-JPG** (radiographs), joined into 44,701 note–report–radiograph records.
- Split **at the patient level** into training, validation and test sets (80:10:10), so no patient appears
  in more than one set. Results are reported on a fixed set of **1,000 test records**.
- **Discharge_Me** is used only for external evaluation, without retraining: each record is joined to a
  chest radiology report from the same admission and a chest radiograph of the same patient (1,000 records).

### 6.2 Measures

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

## 7. Data access

The MIMIC datasets and Discharge_Me are available on [PhysioNet](https://physionet.org/) after
credentialing and signing the data use agreement. No source patient data (notes, reports or images) is included in this repository.

---

## 8. Repository structure and code

```
.
├── README.md                          # this page: supplementary material and usage
├── LICENSE                            # MIT License
└── src/
    ├── data/                          # data preparation
    │   ├── 01_build_multimodal_join.py    # join notes, reports and radiographs into records
    │   ├── 02_split_dataset.py            # patient-level 80:10:10 split
    │   ├── 04_preprocess.py               # image resizing (512 x 512) and text normalization
    │   └── 04b_clean_text.py              # placeholder removal and note truncation
    └── codes/                         # training, generation and evaluation
        ├── models_config.py               # model id and output folders
        ├── 05_multimodal_dataset.py       # dataset (note, report and radiograph)
        ├── 06_train_lora.py               # LoRA fine-tuning: --mode ft_mm (with radiograph), --mode ft (text-only baseline)
        ├── 08_generate.py                 # three-pass generation (--include-radiology, --include-vqa; --no-image for the baseline)
        └── 10_evaluate.py                 # the eight evaluation measures
```

The scripts in `src/codes/` are the reference implementation for LLaVA-Med-7B; CheXagent-8b and
MedGemma-27B follow the same pipeline with their own model loading and chat format.

Example:

```bash
python src/codes/06_train_lora.py --mode ft_mm --fsdp          # train with the radiograph
python src/codes/06_train_lora.py --mode ft --fsdp             # train the text-only baseline
python src/codes/08_generate.py --mode ft_mm --include-radiology --include-vqa
python src/codes/10_evaluate.py <predictions.jsonl>
```

Each script reads and writes a `data/` folder next to it.

---

## 9. Example generated summary

One summary generated by **LLaVA-Med-7B fine-tuned with the radiograph** (`--mode ft_mm`) for a
de-identified MIMIC test record, shown **unedited** (blank name and age fields are the
de-identification placeholders). The source note, report and image are not included. The output is
shown as generated, including its errors: the radiology findings and the radiograph interpretation
contain statements that do not match this patient.

### Part 1: Brief hospital course

> Mr. is a year old male with a history of Hodgkin's Lymphoma s/p splenectomy who presented with fever and cough.
>
> **Fever/Cough:** Patient presented with fever and cough. He was started on empiric azithromycin by his PCP for presumed PNA. He failed to improve, remained intermittently febrile with cough productive of yellow sputum. He represented to his PCP had a WBC of 14 (N90), CXR w L-sided infiltrate and was changed to moxifloxacin. Patient reported improvement cough, fever curve since starting moxifloxacin. Given his concern for failure to improve he presented to the ED for further evaluation and management. In the ED he received 2L NS and was started on vancomycin and ordered for CTX/flagyl (did not receive prior to leaving ED). On the floor, he remained afebrile and his WBC count trended down. He was discharged on 2 weeks of moxifloxacin.
>
> **Hypothyroidism:** Continued Synthroid 75mcg daily.
>
> **HTN:** Continued Toprol 100mg daily, Norvasc 2.5mg daily, Hctz 12.5mg daily, Terazosin 1mg daily, ASA 81mg daily, Lisinopril 20mg daily.
>
> **Transitional Issues:**
> - Patient will need to complete 2 weeks of moxifloxacin
> - Patient will need to have repeat CXR in weeks to document resolution of LLL opacity

### Part 2: Radiology findings

> Ms. is a year old female with a history of splenectomy who presented with cough and chills. She was admitted to the hospital for observation and treatment of presumed pneumonia. A chest x-ray was obtained which did not show any evidence of pneumonia. She was started on levofloxacin and azithromycin. She remained afebrile with a normal white blood cell count. She was discharged home on a 5 day course of levofloxacin and azithromycin. She will follow up with her primary care provider.

### Part 3: Interpretation of the radiograph

> The chest x-ray shows a large right pleural effusion. There is no evidence of pneumothorax. The patient is also noted to have a widened mediastinum. 1. Pleural effusion: The patient was admitted for further workup of the large right pleural effusion. A thoracentesis was performed and 1.5 L of fluid was drained. The fluid was sent for cytology and the results are pending. The patient was also started on a course of levofloxacin for a possible pneumonia. The patient's chest x-ray showed a decrease in the size of the pleural effusion after the thoracentesis. The patient was discharged with follow-up with her PCP and a repeat chest x-ray. 2. Widened mediastinum: ...

---

## License

[MIT License](LICENSE). The MIMIC data and Discharge_Me are not covered by this license and must be
obtained from PhysioNet under its data use agreement.

## Citation

The citation will be added after publication.
