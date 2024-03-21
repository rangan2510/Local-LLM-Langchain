!apt-get install pciutils #drivers for nvidia
!curl -fsSL https://ollama.com/install.sh | sh #servr
!curl -L -o biomistral.gguf https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/resolve/main/BioMistral-7B.Q8_0.gguf?download=true #model - quantosed -7b parameters
!echo "FROM ./biomistral.gguf" > Modelfile #creation of modelfile

#run ollama serve in terminal 

!ollama create biomistral -f Modelfile #loading modelfile into server

! pip install -q langchain langchain-community faiss-cpu langchain-openai tiktoken
#"unstructured[md]" numpy


import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import MarkdownTextSplitter
markdown_splitter = MarkdownTextSplitter(
    chunk_size = 150,
    chunk_overlap=0
                                         
)


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
def mkd_splitter(file_path):
    os.chdir(file_path)
    list_of_folders = os.listdir()
    markdown_files = []

    for folder_name in list_of_folders:
        folder_path = os.path.join('.', folder_name)

        # Print the folder name
        #print(f"\nFiles in folder '{folder_name}':")

        # List all files in the folder
        files_in_folder = os.listdir(folder_path)
        for file_name in files_in_folder:
            if file_name.endswith('.md') and file_name.lower().startswith(folder_name.lower()):
                #print(file_name)
                markdown_file_path = os.path.join(folder_path, file_name)

                # Open the Markdown file and read its contents
                with open(markdown_file_path, 'r') as file:
                    markdown_text = file.read()

                headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ]

                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_header_splits = markdown_splitter.split_text(markdown_text)
                #print(md_header_splits)

                markdown_files.append(md_header_splits)

    return markdown_files

file_path = '/content/drive/MyDrive/all_drugs_wiki_md'  # Update with the path to your Markdown file

docs_mkd= mkd_splitter(file_path)


docs_mkd_p = [j for i in docs_mkd for j in i]
docs_mkd_p


embeddings = OllamaEmbeddings(model="biomistral")
# %%
# Create a blank vectorstore, then incrementally add embeddings
db = FAISS.from_texts([""], embeddings)
for doc in tqdm(docs_mkd_p):
    db_ = FAISS.from_documents([doc], embeddings)
    db.merge_from(db_)

print(db.index.ntotal)
db.save_local("faiss_index_mkd")

! zip -r faiss_index_mkd.zip faiss_index_mkd #storing the faiss_index file in the form of zip file. 

retriever = db.as_retriever(k=2) #top n results
docs_mkd_p = retriever.invoke("Clinical trials for Abemaciclib") #fetches the data.
docs_mkd_p

