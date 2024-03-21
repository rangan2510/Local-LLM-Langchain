#%%
# Imports
# Split of boilerplate code
#imports needed for embeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#imports needed for chat
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.memory import ChatMessageHistory
from langchain_community.chat_models import ChatOllama

#%%
embeddings = OllamaEmbeddings(model="biomistral")

# %%
# load up saved db as our vectorstore and check retrieved docs
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(k=2)
docs = retriever.invoke("Clinical trials for Abemaciclib")
docs

# %%
#build the chatbot
chat = ChatOllama(model="biomistral")

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
        ),
    ]
)

query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("messages", [])) == 1,
        # If only one message, then we just pass that message's content to retriever
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
    query_transform_prompt | chat | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")

#%%
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

ephemeral_chat_history = ChatMessageHistory()

ephemeral_chat_history.add_user_message("What are the targets of Alpelisib?")

response = conversational_retrieval_chain.invoke(
    {"messages": ephemeral_chat_history.messages},
)

ephemeral_chat_history.add_ai_message(response["answer"])

response

#%%
ephemeral_chat_history.add_user_message("What other drugs have the same target?")

conversational_retrieval_chain.invoke(
    {"messages": ephemeral_chat_history.messages}
)

ephemeral_chat_history.add_ai_message(response["answer"])

# %%
