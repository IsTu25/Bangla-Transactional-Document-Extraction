# Literature Check & Gap Analysis: Bangla Key Information Extraction (KIE)

**Project:** Bangla Transactional-Document Extraction: Span-Grounded JSON with a Verifier-Driven Data Flywheel  
**Date:** October 17, 2026  
**Status:** Completed (Phase 0 Deliverable)

---

## 1. Executive Summary & Landscape Shift

Document Understanding and Key Information Extraction (KIE) in low-resource Indic languages, particularly **Bangla (Bengali)**, have historically lagged behind English, Latin, and CJK benchmarks (e.g., FUNSD, CORD, SROIE, DocVQA). However, recent breakthroughs—most notably **BaFCo** (ECCV 2026) and **Khondo** (2026)—have established strong public baselines for **Bangla administrative and government forms**.

This literature check establishes:
1. **Bangla administrative forms are already covered** by BaFCo and Khondo.
2. **Bangla transactional documents** (utility bills, mobile financial services/MFS receipts, retail shop POS invoices) remain **completely unaddressed**.
3. Existing benchmarks evaluate large multimodal models (MLLMs) purely as static evaluation targets. They provide **no span grounding**, **no automated verification**, and **no continuous training flywheel** for efficient small-model deployment.

---

## 2. In-Depth Benchmark Review

### 2.1 BaFCo: Complex Bangla Form Comprehension
* **Citation:** Abu Tyeb Azad, Ishita Sur Apan, Fahim Ahmed, Sumaiya Karim Katha, Ezharuddin Jubaer, Armun Alam, Pranjal Kumar Nandi, Amin Ahsan Ali, Aman Chadha, Md Mofijul Islam, AKM Mahbubur Rahman. *“BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension.”* arXiv:2607.05614 (Submitted July 6, 2026; Accepted at ECCV 2026).
* **Corpus Scope:** 200 multi-page authentic Bangladeshi government forms (316 pages) across 15 administrative sectors (agriculture, primary/secondary education, civil service banking, land registration, municipal revenue).
* **Annotation Breakdown:**
  * **Document Layout Analysis (DLA):** 16,382 layout entities and 8,771 inter-field relationships annotated with fine-grained bounding boxes across 26 form entity types.
  * **Key Information Extraction (KIE):** 1,926 key-value pairs across 156 forms.
* **Evaluation Focus:** Benchmarked zero-shot and few-shot performance of proprietary and open MLLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Qwen2-VL).
* **Key Omissions & Limitations:**
  * Strictly administrative/government applications (no utility bills, POS receipts, or mobile money transfers).
  * Static evaluation only; does not provide a training pipeline or verifier loop.
  * Lacks span-level character grounding for JSON schema outputs.

### 2.2 Khondo: Document Packet Splitting for Bangla Forms
* **Citation:** Abu Tyeb Azad, Fahim Ahmed, Ishita Sur Apan, Ezharuddin Jubaer, Sumaiya Karim Katha, Armun Alam, Amin Ahsan Ali, Aman Chadha, Md Mofijul Islam, AKM Mahbubur Rahman. *“Khondo: A Multimodal Benchmark for Document Packet Splitting of Bangla Forms.”* arXiv:2607.21780 (July 2026).
* **Corpus Scope:** Vision-native benchmark of 500+ document packets comprising concatenated government dossiers across 14 administrative sectors.
* **Task:** Document Packet Splitting (DPS)—detecting document boundaries, classifying sub-documents, and ordering pages within multi-document PDFs.
* **Relevance:** While essential for macro-level ingestion of government folders, Khondo does not perform field-level information extraction.

### 2.3 BaDLAD: Bengali Document Layout Analysis Dataset
* **Citation:** Shihab et al. (Bengali.AI / BUET).
* **Corpus Scope:** ~33,690+ human-annotated pages covering polygon annotations for 4 primary structural classes: paragraph, text-box, table, and image.
* **Utility in this Project:** Excellent for pre-training bounding-box segmentation or table detectors, but BaDLAD does not have semantic key-value annotations or transactional schema targets.

### 2.4 Bengali OCR Benchmarks (Bengali OCR 1M, NumtaDB, CDeCR)
* **Bengali OCR 1M:** 1,000,000 synthetic and scanned word-level images. Good for character recognition baselines.
* **NumtaDB:** Bengali handwritten digit dataset (85,000+ digits). Critical reference for Bengali numeral variations (`০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯`).
* **Limitations:** Character and digit datasets do not model document semantics, spatial hierarchy, or structural JSON extraction.

---

## 3. Comparative Gap Matrix

| Dimension | BaFCo (ECCV 2026) | Khondo (2026) | BaDLAD (Bengali.AI) | **Our Project (Bangla Doc KIE)** |
| :--- | :--- | :--- | :--- | :--- |
| **Domain** | Government administrative forms | Government dossiers / packets | Books, legal docs, newspapers | **Transactional docs (Utility, MFS, POS)** |
| **Target Task** | Entity classification & KIE | Packet segmentation / boundary | Bounding polygon DLA | **Span-grounded JSON schema extraction** |
| **Input Modality** | Multi-page image / MLLM | Multi-page image | Page image | **OCR text + character offsets (v1)** |
| **Grounding** | Bounding boxes | Document boundaries | Polygonal layout | **Character-level `source_span` + text slice** |
| **Verification** | Human validation | Ground-truth split metric | IoU / mAP | **Automated verifier (Format, Span, Schema)** |
| **Data Loop** | Static benchmark | Static benchmark | Static benchmark | **Verifier-driven flywheel (Mine & Retrain)** |
| **Target Model** | Frontier MLLM evaluation | Layout models | YOLO / Mask R-CNN | **Small deployable SLMs (Qwen2.5 3B-7B QLoRA)** |
| **Out-of-Domain Use**| Primary benchmark | N/A | Pre-training | **OOD Transfer evaluation testbed** |

---

## 4. Key Differentiators of Our Angle

1. **Transactional Focus:** Everyday economic documents in Bangladesh (DPDC electricity bills, bKash/Nagad SMS and app receipts, Shwapno supermarket slips) are processed millions of times daily. Automating these provides direct industrial and consumer utility.
2. **Span Grounding (`source_span`):** Hallucinations in financial and utility workflows cause catastrophic failures. Requiring every extracted value to specify `{ "value": ..., "source_span": {"start": s, "end": e, "text": "..."} }` guarantees provenance and enables automated scoring.
3. **The Verifier as Universal Engine:** The verifier is not merely an offline test metric; it acts as:
   * **Evaluation metric:** Field F1, exact span match, and schema conformance.
   * **Failure miner:** Clusters bad predictions into format errors, wrong spans, missing fields, or hallucinations.
   * **RL Reward:** Provides scalar reward signals for GRPO (Group Relative Policy Optimization) fine-tuning.
4. **Out-of-Domain Transfer Testing:** By testing our trained small model directly on BaFCo forms without retraining, we evaluate whether transactional span-grounding generalizes to unseen bureaucratic administrative forms.
