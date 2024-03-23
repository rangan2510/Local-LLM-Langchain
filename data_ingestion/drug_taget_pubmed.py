#%%
from pymed import PubMed
from tqdm.auto import tqdm
import pandas as pd
import time, random
import google.generativeai as genai
from _secrets import API_KEYS
GOOGLE_API_KEY = API_KEYS["Gemini"]
genai.configure(api_key=GOOGLE_API_KEY)
#%%
pubmed = PubMed()
#%%

df = pd.read_excel("HSMD-drugs.xlsx")
df
#%%
queries = []
for idx, row in df.iterrows():
    query = str(row["Drugs"]).replace("/"," ").lower() + " " + str(row["Gene"]).replace("-","")
    queries.append(query)
queries = list(set(queries))

#%%
data = []
for q in tqdm(queries):
    time.sleep(random.randint(1,5))
    results = pubmed.query(q, max_results=50)
    for article in results:
    # Extract and format information from the article
        article_id = article.pubmed_id
        title = article.title
        # if article.keywords:
        #     if None in article.keywords:
        #         article.keywords.remove(None)
        #     keywords = '", "'.join(article.keywords)
        publication_date = article.publication_date
        abstract = article.abstract

        # Show information about the article
        print(
            f'{title}\n{abstract}\n'
        )
        data.append([article_id, publication_date, title, abstract, q])
# %%
results_df = pd.DataFrame(data, columns = ['ids', 'date', 'title', 'abstract', 'query'])
results_df.to_excel("drug_target_pubmed.xlsx")

#%%
summarized = []
for idx, row in tqdm(results_df.iterrows(), total=len(results_df)):
    title = row['title']
    abstract = row['abstract']
    if abstract!=None:
        in_prompt = "Given the following title and abstract, write a short but detailed single paragraph summary. Focus on the drug, it's target and the outcome. Do not mention \"This study\" in the response, simply talk in geenral. \nTitle: " + title + "\nAbstract:" + abstract
        ids = row["ids"].replace("\n",",")
        date = row['date'].strftime("%Y %B %d")

        max_retries = 3
        retry_count = 0

        model = genai.GenerativeModel('gemini-pro')

        print(idx+1, "Attempting to summarize...")
        while retry_count < max_retries:
            try:
                response = model.generate_content(in_prompt)
                summary_text = response.text
                break
            except Exception as e:
                retry_count += 1
                time.sleep(random.randint(1, 5))
                print(e)
                summary_text = ""

        summarized.append([ids, date, title, abstract, summary_text, row['query']])
        print("Done.")

# %%
summarized_df = pd.DataFrame(summarized, columns = ['ids', 'date', 'title', 'abstract', 'summary_text', 'query'])
summarized_df.to_csv("drug_target_pubmed.tsv", sep="\t")