# Bangla Transactional-Document Extraction: Project Plan
**Span-Grounded JSON Extraction with a Verifier-Driven Data Flywheel**  
**Date:** October 17, 2026 | **Version:** 1.0.0

---

## 1. Project Overview & Motivation

Key Information Extraction (KIE) on low-resource Indic administrative documents received significant academic attention with the publication of **BaFCo** (ECCV 2026) and **Khondo** (2026). While these datasets comprehensively cover Bangladeshi governmental forms, **everyday transactional documents**—such as utility bills, mobile financial service (bKash/Nagad) receipts, and retail supermarket POS slips—remain completely untouched.

Furthermore, traditional document extraction models frequently suffer from hallucinations, brittle parsing, and high inference costs when relying on frontier proprietary MLLMs.

This project delivers:
1. **Span-Grounded Structured Extraction:** Extracts strict JSON adhering to schema where every field provides both its canonical normalized `value` and its character-exact `source_span` (`start`, `end`, `text`).
2. **Universal Automated Verifier:** An automated programmatic scoring engine that verifies format validity, span existence in raw OCR text, value-span alignment, and schema conformance without requiring human-in-the-loop validation.
3. **Data Flywheel:** Clustered failure mining that diagnoses model errors (format, missing, wrong span, hallucination) and generates targeted synthetic data for continuous retraining.
4. **Deployable Small Language Models (SLMs):** QLoRA-tuned 3B–7B models (Qwen 2.5) matching or exceeding frontier LLMs at a fraction of latency and cost.
5. **Out-of-Domain Transfer Validation:** Using BaFCo government forms as an external transfer benchmark to prove zero-shot generalization.

---

## 2. Core Method: Ground, Verify, Mine, Retrain

```
                    ┌────────────────────────────┐
                    │ Raw Bangla Document (OCR)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │    Model Prediction        │
                    │ {value, source_span}       │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │     AUTOMATED VERIFIER     │
                    │ 1. Schema Conformance      │
                    │ 2. Span Existence in OCR   │
                    │ 3. Value-Span Consistency  │
                    └─────────────┬──────────────┘
                                  │
                   Score & Taxonomy Diagnostics
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌──────────────────┐                             ┌──────────────────┐
│  High Verifier   │                             │   Mined Errors   │
│  Score (>=0.95)  │                             │ (Format/Span/    │
└────────┬─────────┘                             │  Hallucination)  │
         │                                       └────────┬─────────┘
         ▼                                                │
┌──────────────────┐                                      ▼
│  Trusted Silver  │                             ┌──────────────────┐
│  Data Pool / RL  │                             │ Targeted Synthetic│
│  Reward (GRPO)   │                             │ Data Generation  │
└────────┬─────────┘                             └────────┬─────────┘
         │                                                │
         └─────────────────► RETRAIN ◄────────────────────┘
```

### The 5 Operational Steps:
1. **Ground:** Fine-tune the base model so every output field provides `{ "value": ..., "source_span": {"start": s, "end": e, "text": "..."} }`.
2. **Verify:** Check that:
   - The JSON adheres strictly to the document schema.
   - The span `[start:end]` exactly slices the raw OCR text.
   - The normalized `value` is mathematically or semantically faithful to the span text.
3. **Mine:** Automatically cluster failures into the Error Taxonomy:
   - `FORMAT_ERROR`: Malformed JSON or syntax failure.
   - `SCHEMA_VIOLATION`: Type mismatch or missing required key.
   - `WRONG_SPAN`: Offsets do not match sliced text.
   - `VALUE_SPAN_MISMATCH`: Span text disagrees with normalized value.
   - `HALLUCINATED`: Value generated without grounding.
   - `MISSING_FIELD`: Document ground truth field omitted.
4. **Generate:** Synthesize targeted training instances focused specifically on mined failure categories.
5. **Retrain:** Apply iterative QLoRA Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) using verifier score directly as the scalar reward.

---

## 3. Project Phases & Timeline

| Phase | Title | Key Objective | Primary Output | Duration |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Lit Check & Gap** | Confirm gap, audit BaFCo & Khondo | `docs/literature_review.md` | 2 days |
| **1** | **Scope & Schemas** | Lock 3 doc types and JSON schemas | `schemas/*.json`, `docs/annotation_guide.md` | 3 days |
| **2** | **Data Pipeline** | 300 real test docs + synthetic generator | `data/test/`, `data/synthetic/` | 2 weeks |
| **3** | **Verifier & Harness**| Automated scoring & error taxonomy | `eval/verifier.py`, `eval/metrics.py` | 1 week |
| **4** | **Baselines** | Zero/few-shot & constrained decoding | `results/summary_table_v0.json` | 3 days |
| **5** | **Fine-Tuning** | QLoRA SFT on Qwen2.5-3B/7B | Model weights v1, check-in log | 1 week |
| **6** | **Flywheel Loop** | Mine failures, generate targeted data, RL | Flywheel improvement curves | 1–2 weeks |
| **7** | **Ship & Artifacts** | FastAPI serving, demo app, HF release | Live demo, HuggingFace repo, paper | 1 week |

---

## 4. Key Early Technical Decisions

| Decision | Selected Option | Rationale |
| :--- | :--- | :--- |
| **Input Modality** | **OCR Text First** (Layout-aware ablation later) | Drastically reduces VRAM footprint, fast iteration, QLoRA-friendly. |
| **OCR Engine** | **Dual Benchmark: EasyOCR & PaddleOCR** | Allows treating OCR character error rate (CER) as an explicit variable. |
| **Base Model** | **Qwen 2.5 (3B / 7B)** | Optimal Bangla tokenization efficiency (1 token/word vs 5+ on LLaMA-3). |
| **Data Strategy** | **Synthetic for Train; Real for Test** | Protects user privacy while providing diverse layout/noise permutations. |
| **Annotation** | **Label Studio + Pre-annotation** | Accelerates gold set creation while enforcing strict span alignment. |

---

## 5. Evaluation Plan & Metrics

* **Field F1 (Normalized):** Macro-averaged precision and recall across normalized values.
* **ANLS (Average Normalized Levenshtein Similarity):** OCR-tolerant string similarity metric standard in document VQA.
* **Span Grounding Rate:** Percentage of extracted fields with 100% exact substring match in the raw OCR input.
* **Schema Validity Rate:** Proportion of outputs passing JSON Schema validation.
* **Hallucination Rate:** Frequency of non-null values lacking verifiable source spans.
* **Out-of-Domain Transfer (BaFCo):** Zero-shot transfer score on authentic government forms.
