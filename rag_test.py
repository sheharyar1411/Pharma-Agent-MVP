import json
import time
import os
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

client = Groq()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading and Chunking Medical Dataset...")
with open('clinical_database.json', 'r') as file:
    medical_data = json.load(file)

documents = []
for item in medical_data:
    content = (
        f"Disease: {item['disease']}. "
        f"Category: {item['disease_category']}. "
        f"Symptoms: {', '.join(item['all_symptoms'])}. "
        f"Recommended Drug: {item['generic_drug']}. "
        f"Serious Side Effects: {item['serious_side_effect']}. "
        f"Pregnancy Safe: {item['pregnancy_safe']}. "
        f"Drug Interaction Risk: {item['drug_interaction_risk']}."
    )
    
    doc = Document(page_content=content, metadata={
        "drug": item['generic_drug'], 
        "disease": item['disease']
    })
    documents.append(doc)

print("Building Vector Database...")
vectorstore = Chroma.from_documents(documents, embeddings)

user_query = "I am a pregnant woman experiencing severe joint pain and a migraine. I am currently taking blood pressure medication, so I am worried about drug interactions. What generic medicine should I take?"
print(f"\nUser Query: {user_query}")
print("Searching Database and Generating Response via Groq...\n")

start_time = time.time()

# Retrieval Phase
retrieved_docs = vectorstore.similarity_search(user_query, k=2)
retrieved_context = "\n".join([doc.page_content for doc in retrieved_docs])

# Generation Phase
system_prompt = f"""You are an expert clinical pharmacologist. 
Recommend a safe generic formula based strictly on the retrieved medical context below.
Pay close attention to user allergies. If a drug is contraindicated for the user, do not recommend it.

RETRIEVED MEDICAL CONTEXT:
{retrieved_context}
"""

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ],
    temperature=0.1
)

end_time = time.time()

print(f"🤖 AI Pharma Agent:\n{response.choices[0].message.content}\n")
print(f"⏱️ Generation Time: {round(end_time - start_time, 2)} seconds")