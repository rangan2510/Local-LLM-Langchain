#%%
# Split of boilerplate code
# Includes only data ingestion to pickle file
import os
from tqdm.auto import tqdm
from typing import Dict

#imports needed for embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

#%%
# Ingestion
##  select all markdown files that are wikipedia articles
selected_files = []
for root, dirs, files in os.walk(r"data"):
    for file in files:
        selected_files.append(os.path.join(root, file))
selected_files
#%%
docs = []
for markdown_path in selected_files:
    loader = TextLoader(markdown_path)
    data = loader.load()
    docs += data

embeddings = OllamaEmbeddings(model="biomistral")

# %%
# Create a blank vectorstore, then incrementally add embeddings
db = FAISS.from_texts([""], embeddings)
for doc in tqdm(docs):
    db_ = FAISS.from_documents([doc], embeddings)
    db.merge_from(db_)

print(db.index.ntotal)
db.save_local("faiss_index")
# %%
