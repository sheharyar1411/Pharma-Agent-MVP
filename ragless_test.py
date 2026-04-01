import json
import time
import os
from groq import Groq

# Initialize Groq client (it automatically picks up the GROQ_API_KEY environment variable)
client = Groq()

print("Loading medical dataset...")
with open('clinical_database.json', 'r') as file:
    medical_data = json.load(file)

dataset_string = json.dumps(medical_data, indent=2)

system_prompt = f"""You are an expert clinical pharmacologist. 
Your task is to recommend a generic pharmaceutical formula based on the user's symptoms.
You MUST ONLY use the provided dataset below. Do not use outside knowledge.
If a user mentions an allergy or contraindication, ensure your recommendation is safe by suggesting an alternative from the list.

DATASET:
{dataset_string}
"""

user_query = "I am a pregnant woman experiencing severe joint pain and a migraine. I am currently taking blood pressure medication, so I am worried about drug interactions. What generic medicine should I take?"
print(f"\nUser Query: {user_query}")
print("Generating response via Groq... (Tracking TTFT)\n")

start_time = time.time()

# Call Llama 3.3 70B via Groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ],
    temperature=0.1 # Low temperature for medical accuracy
)

end_time = time.time()

print(f"🤖 AI Pharma Agent:\n{response.choices[0].message.content}\n")
print(f"⏱️ Generation Time: {round(end_time - start_time, 2)} seconds")