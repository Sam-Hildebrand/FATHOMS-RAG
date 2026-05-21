# FATHOMS-RAG: A Framework for the Assessment of Thinking and Observations in Multimodal Systems that use Retrieval Augmented Generation

FATHOMS-RAG is a lightweight benchmark for evaluating end-to-end Retrieval-Augmented Generation (RAG) pipelines in multimodal settings. Unlike component-level benchmarks, this framework evaluates the full pipeline: document ingestion, retrieval, multimodal reasoning, answer generation, and hallucination behavior.

All evaluation outputs are available in the `Results/` folder.

---

## Overview

RAG systems are often evaluated on retrieval quality or generation quality in isolation. FATHOMS-RAG evaluates the system as a whole. It focuses specifically on multimodal scientific documents where key information may be encoded in text, tables, figures, or combinations of these across multiple papers.

The dataset contains 93 human-curated questions derived from eight open-access AI and machine learning papers (arXiv, CC-BY). Questions are organized into five categories:

- Text-only  
- Tables  
- Images  
- Multimodal (within a single document)  
- Cross-document multimodal  

These categories are designed to expose weaknesses in ingestion, retrieval, and reasoning—particularly when structured or visual information is involved.

---

## Evaluation Method

Correctness is measured using phrase-level recall. Each question includes one or more acceptable answers defined by required key phrases. A model receives partial credit based on how many required phrases appear in its response, with the maximum score taken across acceptable variations. Scores range from 0.0 to 1.0.

Hallucination detection is handled separately. We use a nearest-neighbor embedding classifier to distinguish between:

- Statement-style answers  
- Abstention-style answers  

If a response is classified as a statement and does not achieve full phrase-level recall, it is labeled as a hallucination. Abstentions are not considered hallucinations. This method works without access to token probabilities or model internals, making it applicable to both open-source and closed-source systems.

A small third-party human evaluation study showed strong agreement with the automatic metrics (average 4.62/5 for correctness and 4.53/5 for hallucination detection).

---

## Pipelines Evaluated

We evaluated three types of systems:

1. A text-only RAG pipeline built with LlamaIndex.
2. A layout- and OCR-aware pipeline using Docling with EasyOCR.
3. Closed-source multimodal APIs (Claude Sonnet 4, Gemini 2.5 Flash, GPT-4.1, GPT-4o).

Text-only pipelines perform moderately on pure text questions but fail on tables and images, with high hallucination rates. OCR-based ingestion improves multimodal coverage and reduces hallucination, but structured table reasoning remains weak. Closed-source models substantially outperform open-source pipelines overall, yet all systems struggle with cross-document multimodal reasoning.

Cross-document multimodal reasoning remains the most difficult category across every evaluated system.

---

## Key Findings

- Text-only RAG pipelines are insufficient for multimodal documents.
- OCR and layout-aware ingestion significantly improve performance.
- Closed-source systems achieve higher correctness and lower hallucination rates.
- Cross-document multimodal reasoning is a consistent bottleneck.

---

## Limitations

Thish is in-progress work and the current dataset is very small. While this allows for very fast evaluation, it does not yet allow the granularity that may be required in some applications. Also while suitable for scientific PDFs, it may not generalize to other domains without expansion. Phrase-level matching may underestimate semantic correctness, and the hallucination classifier depends on a limited annotated training set. Closed-source systems may also rely on internal retrieval mechanisms that are not visible.

---

## Repository Structure

- `data/` — Question and answer definitions  
- `evaluation/` — Scoring and hallucination detection code  
- `Results/` — Model outputs and aggregated metrics

## Attribution for Papers Queried About

The following papers were used in the construction of the preliminary dataset. All papers below are CC-BY (Attribution 4.0) Licensed.

```
@misc{zhmoginov2022hypertransformermodelgenerationsupervised,
      title={HyperTransformer: Model Generation for Supervised and Semi-Supervised Few-Shot Learning}, 
      author={Andrey Zhmoginov and Mark Sandler and Max Vladymyrov},
      year={2022},
      eprint={2201.04182},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2201.04182}, 
}
```

```
@misc{zhang2024multimodalchainofthoughtreasoninglanguage,
      title={Multimodal Chain-of-Thought Reasoning in Language Models}, 
      author={Zhuosheng Zhang and Aston Zhang and Mu Li and Hai Zhao and George Karypis and Alex Smola},
      year={2024},
      eprint={2302.00923},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2302.00923}, 
}
```

```
@misc{wang2023docllmlayoutawaregenerativelanguage,
      title={DocLLM: A layout-aware generative language model for multimodal document understanding}, 
      author={Dongsheng Wang and Natraj Raman and Mathieu Sibue and Zhiqiang Ma and Petr Babkin and Simerjot Kaur and Yulong Pei and Armineh Nourbakhsh and Xiaomo Liu},
      year={2023},
      eprint={2401.00908},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.00908}, 
}
```

```
@misc{li2025nphardeval4vdynamicevaluationlarge,
      title={NPHardEval4V: Dynamic Evaluation of Large Vision-Language Models with Effects of Vision}, 
      author={Xiang Li and Wenyue Hua and Kaijie Zhu and Lingyao Li and Haoyang Ling and Jinkui Chi and Qi Dou and Jindong Wang and Yongfeng Zhang and Xin Ma and Lizhou Fan},
      year={2025},
      eprint={2403.01777},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.01777}, 
}
```

```
@misc{oesch2024pathautonomouscyberdefense,
      title={The Path To Autonomous Cyber Defense}, 
      author={Sean Oesch and Phillipe Austria and Amul Chaulagain and Brian Weber and Cory Watson and Matthew Dixson and Amir Sadovnik},
      year={2024},
      eprint={2404.10788},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2404.10788}, 
}
```

```
@misc{yang2025chartmimicevaluatinglmmscrossmodal,
      title={ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation}, 
      author={Cheng Yang and Chufan Shi and Yaxin Liu and Bo Shui and Junjie Wang and Mohan Jing and Linran Xu and Xinyu Zhu and Siheng Li and Yuxiang Zhang and Gongye Liu and Xiaomei Nie and Deng Cai and Yujiu Yang},
      year={2025},
      eprint={2406.09961},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2406.09961}, 
}
```

```
@misc{wang2025enigmaevalbenchmarklongmultimodal,
      title={EnigmaEval: A Benchmark of Long Multimodal Reasoning Challenges}, 
      author={Clinton J. Wang and Dean Lee and Cristina Menghini and Johannes Mols and Jack Doughty and Adam Khoja and Jayson Lynch and Sean Hendryx and Summer Yue and Dan Hendrycks},
      year={2025},
      eprint={2502.08859},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2502.08859}, 
}
```

```
@misc{utkarsh2025physicsconstrainedflowmatchingsampling,
      title={Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints}, 
      author={Utkarsh Utkarsh and Pengfei Cai and Alan Edelman and Rafael Gomez-Bombarelli and Christopher Vincent Rackauckas},
      year={2025},
      eprint={2506.04171},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.04171}, 
}
```

# Cite

Please use the following BibTex citation when referencing this work:
```
@misc{hildebrand2025fathomsragframeworkassessmentthinking,
      title={FATHOMS-RAG: A Framework for the Assessment of Thinking and Observation in Multimodal Systems that use Retrieval Augmented Generation}, 
      author={Samuel Hildebrand and Curtis Taylor and Sean Oesch and James M Ghawaly Jr and Amir Sadovnik and Ryan Shivers and Brandon Schreiber and Kevin Kurian},
      year={2025},
      eprint={2510.08945},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.08945}, 
}
```
