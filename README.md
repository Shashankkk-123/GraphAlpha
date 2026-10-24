# GraphAlpha
# 🕸️ Local GraphRAG & Entity Resolution Pipeline

![Neo4j](https://img.shields.io/badge/Neo4j-018bff?style=flat&logo=neo4j&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-black?style=flat)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

## 📖 Abstract
This project is a proof-of-concept **GraphRAG (Knowledge Graph Augmented Generation)** pipeline for processing unstructured financial disclosures using local, open-weights LLMs. It explores two concrete failure modes that show up when running knowledge-graph extraction on a small (7B) local model instead of a large hosted one — **entity resolution blindspots** and **information loss during structured extraction** — and documents the fixes attempted for each, including their limitations.

## 🚀 Key Milestones & Findings

### 1. The Acronym Blindspot (Entity Resolution)
Lexical matching (Levenshtein, threshold 0.80) found **zero** matches between un-normalized entity names like "SVB" and "Silicon Valley Bank" — the string similarity between them is too low for edit-distance matching to help.

Switching to **semantic (embedding-based) clustering** did better, but the right threshold wasn't obvious upfront. At cosine similarity 0.85, only one pair merged. Diagnosing specific pairs showed "Silicon valley bank" / "Svb financial group" scored 0.719 — below the 0.85 cutoff. Lowering the threshold to 0.71 did correctly merge the SVB pair, **but in the same run it also incorrectly merged two unrelated entities, "Apple inc." and "Microsoft corp" (score 0.720)**. The two problems sit on either side of the exact same threshold — there's no single cutoff in this experiment that resolves true acronym pairs without also risking false merges between unrelated entities. (Separately, node IDs returned by Neo4j/APOC after a merge come back as a list rather than a string, which needed a small type-check fix before clustering could run on them.)

### 2. Two Distinct Extraction Failures, Diagnosed Separately
Two different problems showed up at two different stages, and they needed two different fixes:

- **Schema-constrained extraction was unstable.** Forcing the 7B model to extract strictly into a fixed `<Subject> <Predicate> <Object>` schema (`SchemaLLMPathExtractor`) caused unstable relationship generation. Switching to an unconstrained path extractor (`SimpleLLMPathExtractor`) produced a more stable graph.
- **The retriever was only returning compressed triples, not source text.** Inspecting retrieved nodes directly showed the retriever surfacing extracted facts like `Apple inc. -> Located in -> Asia` instead of the original sentence — context useful for graph traversal, but too lossy for open-ended Q&A. Rebuilding the retriever with `VectorContextRetriever(..., include_text=True)` restored access to the original text alongside the graph relations.

### 3. Small-Model Generation Loops
The local 7B model, when given a default retrieval-augmented prompt, fell into repetitive self-dialogue loops (generating its own fake "Query: ... Answer: ..." turns instead of stopping). A minimal custom prompt template, explicitly instructing the model to answer the query directly from context, resolved this for subsequent queries.

---

## 🛠️ Methodology & Architecture

* **LLM Engine:** `Qwen/Qwen2.5-7B-Instruct` (Quantized to 4-bit via `bitsandbytes` for consumer GPU hardware).
* **Embeddings:** `BAAI/bge-small-en-v1.5` for vector search and semantic clustering.
* **Orchestration:** LlamaIndex (Core, Neo4j, HuggingFace integrations) — `SimpleLLMPathExtractor` for graph construction, `VectorContextRetriever` for retrieval.
* **Graph Database:** Neo4j (AuraDB Cloud).
* **Algorithms:** Levenshtein Distance (Lexical Resolution), Cosine Similarity (Semantic Resolution).

---

## 💻 Output Examples: Entity Bridging via Semantic Resolution

**Prompt:** *"What specific economic pressures heavily impacted SVB's capital ratios?"*

**Response:**
> "Interest rate risks heavily impacted Silicon Valley Bank's capital ratios..."

The query refers to the entity only as "SVB," while the underlying document text uses "Silicon Valley Bank." The system correctly bridges the two, reflecting the semantic entity-resolution step described above.

*Note: not every retrieval gap was fully closed — a separate query asking about Apple's supply chain issues returned a "context does not provide this information" response even after the retriever fix, since the source documents didn't contain that detail in the first place. `include_text=True` fixed the context-loss problem but doesn't manufacture information that was never extracted or present in the corpus.*

---

## 🔮 Future Work: Cross-Domain Scale & Automated Thresholds
This architecture establishes a baseline for local GraphRAG experimentation. Future work will extend this pipeline to investigate **Dynamic Thresholding**—utilizing a secondary local classification model to automatically adjust semantic merging thresholds at runtime based on the specific industry domain (e.g., Biomedical vs. Financial text), aiming to reduce the false-positive merges observed with a fixed threshold.

---

## ⚙️ How to Run

1. Open the `.ipynb` notebook in Google Colab or your local Jupyter environment.
2. Set Runtime to **T4 GPU** (or equivalent 16GB+ VRAM hardware).
3. Ensure you have an active **Neo4j AuraDB** instance and a **Hugging Face** API token.
4. Install dependencies: `pip install llama-index llama-index-graph-stores-neo4j neo4j transformers bitsandbytes`
5. Run the cells sequentially to ingest the corpus, execute resolution algorithms, and test the Hybrid Retriever.