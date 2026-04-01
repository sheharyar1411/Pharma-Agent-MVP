# Pharma Agent MVP: RAG vs. RAG-less Architecture R&D

## Project Overview
This document outlines the research and development (R&D) blueprint for evaluating two distinct AI architectures: **Retrieval-Augmented Generation (RAG)** and **RAG-less (Long-Context Native)**. The objective is to build a medical AI agent that recommends **generic pharmaceutical formulas** based on user-described symptoms or diseases, strictly avoiding commercial brand bias.

By comparing these approaches, we can determine the optimal balance of **latency, accuracy, and efficiency** for medical data processing. Orchestrating a multi-agent router for symptom classification here will naturally bridge the frontend and backend components, similar to building comprehensive full-stack applications.

---

## Dataset Selection (MVP Phase)
For this MVP, the dataset must be dense enough for meaningful retrieval but compact enough to fit inside a single local LLM context window.

* **Primary Source:** WHO Model List of Essential Medicines (EML) in JSON/CSV format.
* **Alternative Source:** Disease-Symptom-Treatment datasets (Kaggle).
* **Data Preparation:** * Remove extraneous columns (e.g., historical dates, internal IDs).
    * Standardize the format to clear Markdown or structured JSON to reduce token overhead.

---

## Architecture 1: The RAG Pipeline (Grounded Retrieval)
This approach relies on extracting only the most relevant snippets of medical data to answer the user's query, ensuring high traceability and lower token usage per inference.

### 1. Requirements
* **LLM Engine:** Llama 3.1 (8B) or Llama 4 (8B) via Ollama.
* **Embedding Model:** `all-MiniLM-L6-v2` or `BGE-m3` (via HuggingFace).
* **Vector Database:** ChromaDB or FAISS (Local & Free).
* **Orchestration Framework:** LangChain or LlamaIndex.

### 2. Architecture Flow
1.  **Ingestion & Chunking:** The medical dataset is divided into semantic chunks (e.g., one chunk per disease/formula pair).
2.  **Vectorization:** The embedding model converts these chunks into dense vectors and stores them in ChromaDB.
3.  **Retrieval:** The user inputs a symptom. The system converts this query into a vector and retrieves the top-*k* (e.g., top 3) most semantically similar chunks.
4.  **Generation:** The retrieved chunks are injected into the LLM's prompt. The LLM synthesizes a response recommending the generic formula.

> **Pro-Tip:** Medical RAG thrives on exact keyword matching. Implementing a **Hybrid Search** (combining BM25 for exact chemical names with Vector Search for symptom descriptions) drastically improves retrieval accuracy.

### 3. Pros and Cons
| Metric | Advantage | Disadvantage |
| :--- | :--- | :--- |
| **Token Usage** | Highly efficient; only sends ~500 tokens to the LLM. | Requires overhead to generate and store embeddings. |
| **Speed (TTFT)** | Fast generation once chunks are retrieved. | Extra latency added during the vector search phase. |
| **Traceability** | Excellent; easily cites the exact chunk used. | Susceptible to retrieving the wrong chunk if the search fails. |

---

## Architecture 2: The RAG-less Pipeline (Long-Context Native)
This approach bypasses the vector database entirely, leveraging the expanded context windows of modern models to read the entire medical dataset during every query.

### 1. Requirements
* **LLM Engine:** Llama 3.1 (8B) or Llama 4 (8B) via Ollama.
* **Context Capacity:** Must support at least 100k+ tokens natively.
* **Orchestration:** Python (Standard prompt formatting).

### 2. Architecture Flow
1.  **Data Condensation:** The entire dataset is formatted into a dense, clean text block.
2.  **The Mega-Prompt:** The dataset is injected directly into the system prompt alongside strict instructions: *"You are a clinical pharmacologist. Based ONLY on the following dataset, recommend a generic formula..."*
3.  **Generation:** The LLM scans the entire dataset in its memory buffer to find the correct correlation and generates the response natively.

### 3. Pros and Cons
| Metric | Advantage | Disadvantage |
| :--- | :--- | :--- |
| **Infrastructure** | Zero moving parts; no vector DB or chunking strategy needed. | Requires a local machine with sufficient RAM to hold the large context. |
| **Holistic Reasoning** | Can connect symptoms across the entire dataset flawlessly. | Higher risk of the "Lost in the Middle" phenomenon. |
| **Latency** | No retrieval delay. | High Time-to-First-Token (TTFT) due to processing the massive prompt. |

---

## Evaluation Metrics (The R&D Core)
To mathematically determine the superior approach, track these metrics during testing:

1.  **Latency (Time-to-First-Token):** Measure the exact millisecond delay from the user pressing "Enter" to the first generated word.
2.  **Context Efficiency (Cost Simulation):** Log the total input and output tokens for every query to simulate production API costs.
3.  **Hallucination Rate:** Run 50 predefined complex symptom queries through both pipelines and evaluate accuracy.
4.  **Constraint Adherence:** Test edge cases (e.g., *"Patient is allergic to X"*). Does the RAG retrieve the allergy warning? Does the RAG-less model spot it in the massive text?
