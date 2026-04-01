from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import ollama

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create vector store
texts = ["Paracetamol is used for fever.", "Ibuprofen is used for pain."]
db = FAISS.from_texts(texts, embeddings)

# Query
query = "What is paracetamol used for?"
docs = db.similarity_search(query)

# Ask LLM
response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": docs[0].page_content}]
)

print(response['message']['content'])