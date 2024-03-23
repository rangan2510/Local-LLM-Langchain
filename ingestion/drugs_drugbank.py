#%%
import os, time, random
from tqdm.auto import tqdm
from typing import Dict

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

import pandas as pd
#%%
SUMMARIZE = True
GOOGLE_API_KEY = ""

#%%
source_path = '/content/drive/MyDrive/drug_bank.xlsx'
drugbank_df = pd.read_excel(source_path)

drugbank_df = drugbank_df.iloc[: , 1:]
drugbank_df.isnull().sum()
for col in drugbank_df.columns:
    if drugbank_df[col].isnull().sum() > 0:
        drugbank_df[col].fillna(value=" " " ", inplace=True)

list_of_cols= ['description','indication','pharmacodynamics','mechanism-of-action','toxicity','metabolism','absorption',
               'half-life','route-of-elimination']

#%%
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

docs = []

for index, row in drugbank_df.iterrows():
    drug_name = row['name']
    #row_context = ""
    for column in list_of_cols:
        info= str(row[column])
        in_prompt = "Write a single paragraph about " + column + " of  " + drug_name + "based on the following information for breast cancer:\n" + info

        max_retries = 3
        retry_count = 0

        print(index+1, ": attempting to summarize...")
        while retry_count < max_retries:
            try:
                response = model.generate_content(in_prompt)
                summary_text = response.text
                print(summary_text)

                d =  Document(page_content=summary_text, metadata={"source": drug_name,"description": column})
                docs.append(d)
                break
            except Exception as e:
                retry_count += 1
                time.sleep(random.randint(5, 15))
                print(e)
                summary_text = str(e)

    #docs.append(row_context.strip())
    print("Done.")
    docs_ = docs
docs
#%%
embeddings = OllamaEmbeddings(model="biomistral")
# %%
# Create a blank vectorstore, then incrementally add embeddings
db = FAISS.from_texts([""], embeddings)
for doc in tqdm(docs):
    db_ = FAISS.from_documents([doc], embeddings)
    db.merge_from(db_)

print(db.index.ntotal)
db.save_local("faiss_index_summarised_drug_bank_metadata_changed")
# %%
