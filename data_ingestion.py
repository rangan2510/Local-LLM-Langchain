#%%
# Split of boilerplate code
# Includes only data ingestion to pickle file
import os, time, random
from tqdm.auto import tqdm
from typing import Dict

#imports needed for embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document



SUMMARIZE = True
GOOGLE_API_KEY = 'AIzaSyACnLpyEQXYjGy_m0fuWuLG0KSE21PmFno'

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

#%%
# Summarize
if SUMMARIZE:
    import google.generativeai as genai
    
    _docs = []

    for i, doc in enumerate(docs):
        in_prompt = "Summarize the following in a paragraph while preserving details as much as possible: \n" + doc.page_content

        max_retries = 3
        retry_count = 0

        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')

        print(i+1, ": attempting to summarize...")
        while retry_count < max_retries:
            try:
                response = model.generate_content(in_prompt)
                summary_text = response.text
                print(summary_text)
                d =  Document(page_content=summary_text, metadata={"source": "local"})
                _docs.append(d)
                break
            except Exception as e:
                retry_count += 1
                time.sleep(random.randint(5, 15))
                print(e)
                summary_text = str(e)

        print("Done.")

    docs = _docs
# %%
# Create a blank vectorstore, then incrementally add embeddings
db = FAISS.from_texts([""], embeddings)
for doc in tqdm(docs):
    db_ = FAISS.from_documents([doc], embeddings)
    db.merge_from(db_)

print(db.index.ntotal)
db.save_local("faiss_index")
# %%



