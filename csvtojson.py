import pandas as pd
import json
import numpy as np

# 1. Load the dataset (Replace with your actual CSV file name)
print("Loading CSV...")
df = pd.read_csv('Healthcare SymptomDiseaseDrug Research Dataset.csv') # <-- UPDATE THIS FILENAME

# 2. Combine symptom columns into a single clean list, ignoring NaN values
print("Consolidating symptoms...")
symptom_cols = ['symptom_1', 'symptom_2', 'symptom_3', 'symptom_4', 'symptom_5']
df['all_symptoms'] = df[symptom_cols].apply(
    lambda x: [str(i).strip() for i in x if pd.notna(i) and str(i).strip() != ''], axis=1
)

# 3. Select only the crucial columns for the AI's clinical reasoning
cols_to_keep = [
    'disease', 
    'disease_category', 
    'all_symptoms', 
    'generic_drug', 
    'serious_side_effect', 
    'pregnancy_safe', 
    'drug_interaction_risk'
]
df_clean = df[cols_to_keep].copy()

# 4. Drop duplicates to keep the knowledge base lean and efficient
# We only want unique combinations of diseases, symptoms, and drugs
print(f"Original row count: {len(df_clean)}")
df_clean = df_clean.drop_duplicates(subset=['disease', 'generic_drug', 'serious_side_effect']).reset_index(drop=True)
print(f"Optimized row count: {len(df_clean)}")

# 5. Convert to a list of dictionaries for the LLM pipelines
records = df_clean.to_dict(orient='records')

# 6. Save to the JSON file expected by your RAG and RAG-less scripts
with open('clinical_database.json', 'w') as f:
    json.dump(records, f, indent=2)

print("Data successfully cleaned and saved to clinical_database.json!")