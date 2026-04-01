# Pharma Agent MVP

A research and development initiative to compare **Retrieval-Augmented Generation (RAG)** versus **RAG-less (Long-Context Native)** approaches for building an intelligent pharmaceutical recommendation agent that suggests generic medicines based on patient symptoms and diseases.

## 🎯 Project Objective

Build a medical AI agent that:
- Recommends **generic pharmaceutical formulas** based on user-described symptoms or diseases
- Avoids commercial brand bias
- Optimizes for **latency, accuracy, and efficiency**
- Evaluates two distinct architectures: RAG and RAG-less approaches

## 🏗️ Architecture Comparison

### Architecture 1: RAG Pipeline (Grounded Retrieval)
Extracts relevant snippets from a medical database and uses them to ground LLM responses.

**Key Technologies:**
- LLM: Llama 3.1 (8B) via Ollama
- Embedding Model: `all-MiniLM-L6-v2` or `BGE-m3`
- Vector Database: ChromaDB or FAISS
- Framework: LangChain or LlamaIndex

**Advantages:**
- Highly efficient token usage (~500 tokens per inference)
- Excellent traceability (cites exact sources)
- Fast generation once chunks retrieved

**Disadvantages:**
- Requires embedding generation and vector storage overhead
- Extra latency from vector search phase
- Susceptible to retrieval failures

### Architecture 2: RAG-less Pipeline (Long-Context Native)
Injects the entire medical dataset into the LLM's context window for native processing.

**Key Technologies:**
- LLM: Llama 3.1 (8B) via Ollama (100k+ context window)
- Framework: Python with prompt formatting

**Advantages:**
- Zero infrastructure (no vector DB)
- Holistic reasoning across entire dataset
- No retrieval latency

**Disadvantages:**
- Higher RAM requirements
- Long Time-to-First-Token (TTFT)
- Risk of "Lost in the Middle" phenomenon

## 📊 Evaluation Metrics

1. **Latency (TTFT):** Time from query to first generated token
2. **Context Efficiency:** Total input/output tokens per query
3. **Hallucination Rate:** Accuracy on 50 predefined complex symptom queries
4. **Constraint Adherence:** Edge case handling (allergies, contraindications, etc.)

## 📁 Project Structure

```
pharma_agent_mvp/
├── rag_test.py                           # RAG pipeline implementation
├── ragless_test.py                       # RAG-less pipeline implementation
├── test.py                               # Evaluation & testing suite
├── requirements.txt                      # Python dependencies
├── clinical_database.json                # Processed medical dataset (JSON)
├── em1.json                              # WHO Essential Medicines List
├── Healthcare SymptomDiseaseDrug Research Dataset.csv  # Raw dataset
├── csvtojson.py                          # Data conversion utility
├── pharma_agent_architecture_blueprint.md # Detailed architecture documentation
└── README.md                             # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Ollama (with Llama 3.1 8B model)
- 16GB+ RAM (recommended for RAG-less approach)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Pharma-Agent-MVP.git
   cd Pharma-Agent-MVP
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running with Llama 3.1:**
   ```bash
   ollama pull llama2  # or appropriate model
   ollama serve
   ```

### Running Tests

**RAG Pipeline Test:**
```bash
python rag_test.py
```

**RAG-less Pipeline Test:**
```bash
python ragless_test.py
```

**Full Evaluation Suite:**
```bash
python test.py
```

## 📦 Dependencies

### Core LLM & Orchestration
- `langchain` - LLM orchestration
- `langchain-community` - Community integrations
- `ollama` - Local LLM interface

### Embeddings & Vector Search
- `sentence-transformers` - Embedding generation
- `faiss-cpu` - Vector database

### Data Processing
- `pandas` - Data manipulation
- `numpy` - Numerical computing

### Web Framework (Optional)
- `fastapi` - REST API
- `uvicorn` - ASGI server

## 📝 Dataset

- **Primary:** WHO Model List of Essential Medicines (EML)
- **Secondary:** Healthcare Symptom-Disease-Drug Research Dataset
- **Format:** JSON, CSV (converted for optimal efficiency)

All datasets are cleaned and standardized to minimize token overhead.

## 🔍 Key Features

- ✅ Side-by-side RAG vs RAG-less comparison
- ✅ Quantitative metrics (latency, tokens, accuracy)
- ✅ Medical domain-specific data preparation
- ✅ Hybrid search capabilities (BM25 + Vector)
- ✅ Edge case testing (allergies, contraindications)
- ✅ Local-first architecture (no external API calls)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 Documentation

For detailed architecture documentation and design decisions, see [pharma_agent_architecture_blueprint.md](pharma_agent_architecture_blueprint.md).

## ⚠️ Medical Disclaimer

This is an MVP for research purposes. Pharmaceutical recommendations should always be validated by qualified healthcare professionals. This tool is not a substitute for professional medical advice.

## 🐛 Issues & Support

For issues, bugs, or feature requests, please open an issue on GitHub.

---

**Status:** MVP Phase - Active R&D
**Last Updated:** April 2026
