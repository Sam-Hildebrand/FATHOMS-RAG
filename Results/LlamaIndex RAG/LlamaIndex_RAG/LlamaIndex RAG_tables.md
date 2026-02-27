# Overall Scores
| Model | Overall | Overall Hallucinations |
|---|---|---|
| gemma3:12b__LlamaIndex_RAG | 0.28 | 0.52 |
| gemma3:1b__LlamaIndex_RAG | 0.19 | 0.75 |
| gemma3:4b__LlamaIndex_RAG | 0.23 | 0.53 |
| gpt_oss:120b__LlamaIndex_RAG | 0.30 | 0.44 |
| gpt_oss:20b__LlamaIndex_RAG | 0.30 | 0.49 |
| llama3_3:70b__LlamaIndex_RAG | 0.32 | 0.54 |

# Detailed Scores
| Model | Text-Only | Text-Only Hallucination Rate | Tables | Tables Hallucination Rate | Images | Images Hallucination Rate | Multimodal | Multimodal Hallucination Rate | Cross-Document Multimodal | Cross-Document Multimodal Hallucination Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:12b__LlamaIndex_RAG | 0.55 | 0.35 | 0.15 | 0.46 | 0.29 | 0.64 | 0.21 | 0.64 | 0.19 | 0.50 |
| gemma3:1b__LlamaIndex_RAG | 0.36 | 0.48 | 0.08 | 0.85 | 0.07 | 0.86 | 0.13 | 0.84 | 0.29 | 0.70 |
| gemma3:4b__LlamaIndex_RAG | 0.49 | 0.35 | 0.15 | 0.54 | 0.21 | 0.43 | 0.19 | 0.72 | 0.12 | 0.60 |
| gpt_oss:120b__LlamaIndex_RAG | 0.62 | 0.26 | 0.15 | 0.54 | 0.07 | 0.36 | 0.24 | 0.44 | 0.42 | 0.60 |
| gpt_oss:20b__LlamaIndex_RAG | 0.63 | 0.23 | 0.23 | 0.46 | 0.14 | 0.57 | 0.28 | 0.48 | 0.23 | 0.70 |
| llama3_3:70b__LlamaIndex_RAG | 0.62 | 0.26 | 0.15 | 0.38 | 0.21 | 0.64 | 0.34 | 0.60 | 0.29 | 0.80 |
