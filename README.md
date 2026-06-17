# GraphAlpha
# 🕸️ Local GraphRAG & Entity Resolution Pipeline

![Neo4j](https://img.shields.io/badge/Neo4j-018bff?style=flat&logo=neo4j&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-black?style=flat)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

## 📖 Abstract
This project implements a **Knowledge Graph Augmented Generation (GraphRAG)** architecture to process unstructured financial disclosures utilizing local, open-weights Large Language Models (LLMs). This research explores the inherent vulnerabilities in standard Knowledge Graph extraction—specifically Information Extraction Loss and Acronym Blindspots—and implements a robust **Hybrid Graph + Vector Retrieval** system to mitigate them under local edge-hardware constraints.

This project serves as a proof-of-concept for engineering production-grade, hallucination-resistant retrieval pipelines independent of closed-source API models.

## 🚀 Key Milestones & Findings

### 1. The Acronym Blindspot (Entity Resolution)
Initial experiments using Levenshtein Distance (Lexical matching) failed to link un-normalized entities (e.g., "SVB" and "Silicon Valley Bank") due to low string similarity. By implementing **Semantic Vector Clustering**, the system successfully bridged these gaps.
* **Threshold Tuning:** Tuning the Cosine Similarity threshold down to `0.71` successfully merged contextual acronyms.
* **Graph Collapse:** Lowering thresholds below `0.70` caused representation ruin ("Graph Collapse"), erroneously merging distinct entities like *Apple* and *Microsoft*. Array-safe property extraction was engineered to handle these merges dynamically.

### 2. Mitigating Information Extraction Loss
Standard schema-based extractors forced the 7B LLM to drop highly specific contextual phrases (e.g., discarding "supply chain issues") to fit strict `<Subject> <Predicate> <Object>` formats. By transitioning to an `ImplicitPathExtractor`, the pipeline explicitly preserved raw text chunks as a safety net alongside the structural graph nodes.

### 3. Overcoming Small-Model Prompt Choking
When querying dense graph structures, orchestration frameworks inject heavy formatting strings. The local 7B model choked on this syntax, falling into repetitive generation loops. A custom, streamlined prompt template and a `VectorContextRetriever(include_text=True)` were implemented to force the engine to read both relational edges and raw text safely.

---

## 🛠️ Methodology & Architecture

* **LLM Engine:** `Qwen/Qwen2.5-7B-Instruct` (Quantized to 4-bit via `bitsandbytes` for consumer GPU hardware).
* **Embeddings:** `BAAI/bge-small-en-v1.5` for vector search and semantic clustering.
* **Orchestration:** LlamaIndex (Core, Neo4j, HuggingFace integrations).
* **Graph Database:** Neo4j (AuraDB Cloud).
* **Algorithms:** Levenshtein Distance (Lexical Resolution), Cosine Similarity (Semantic Resolution).

---

## 📊 Visualizations & Data Analysis

### Entity Resolution: Semantic Clustering vs. Graph Collapse
*Visual representation demonstrates the 0.71 semantic threshold successfully linking SVB variants, contrasted with the 0.70 threshold incorrectly collapsing distinct tech entities (Apple and Microsoft) into a single corrupted node.*

### Hybrid Retrieval Architecture Flow
*Diagram illustrating the dual-retrieval path: executing Cypher queries for relational structural data while simultaneously computing vector similarities for granular, raw-text fallback to prevent information extraction loss.*

---

## 💻 Output Examples: Hybrid GraphRAG Retrieval

**Prompt:** *"What specific supply chain issues or operational challenges did Apple (AAPL) face in Asia?"*

**Baseline (Naive KG Extractor):**
> "The given context does not provide any specific information about supply chain issues or operational challenges faced by Apple Inc. in Asia. Therefore, based on the provided context, it's not possible to answer this question accurately." 
> 
> *(Note: The data was permanently lost during strict triplet extraction).*

**Steered/Production (Hybrid Retriever + Preserved Text):**
> "The context mentions that Apple Inc. faced supply chain issues without giving further details. To accurately answer this question, more detailed operational information would be needed." 
> 
> *(Note: The hybrid vector fallback successfully bypassed the broken graph structure and retrieved the granular raw text chunk).*

---

## 🔮 Future Work: Cross-Domain Scale & Automated Thresholds
This architecture establishes the baseline for scalable local GraphRAG. Future work will extend this pipeline to investigate **Dynamic Thresholding**—utilizing a secondary local classification model to automatically adjust semantic merging thresholds at runtime based on the specific industry domain (e.g., Biomedical vs. Financial text) to entirely prevent Graph Collapse.

---

## ⚙️ How to Run

1. Open the `.ipynb` notebook in Google Colab or your local Jupyter environment.
2. Set Runtime to **T4 GPU** (or equivalent 16GB+ VRAM hardware).
3. Ensure you have an active **Neo4j AuraDB** instance and a **Hugging Face** API token.
4. Install dependencies: `pip install llama-index llama-index-graph-stores-neo4j neo4j transformers bitsandbytes`
5. Run the cells sequentially to ingest the corpus, execute resolution algorithms, and test the Hybrid Retriever.