#%%writefile app.py
# final one
import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

st.title('Oncologist Co-Pilot')

st.write("#")
st.write("#")

#if "llm" not in st.session_state:
#    st.session_state["llm"] = Ollama(model="f0rodo/bio-mistral-dare")
llm = Ollama(model="f0rodo/bio-mistral-dare")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if ques := st.chat_input("Enter your questions here"):
    st.session_state.messages.append({"role": "user", "content": ques})
    with st.chat_message("user"):
        st.markdown(ques)

    template = f"""You are an AI assistant for oncologists answering questions related to breast cancer, your primary goal is to provide helpful, respectful, and honest assistance. Your answers should be informative and based on factual information from the patient's report or relevant knowledge about genes, drugs, therapies, and clinical trials concerning breast cancer. Please ensure that all information shared is medically sound and factually correct. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    Accordingly answer, {ques}.

    """

    prompt = PromptTemplate(
        input_variables=["ques"], template=template
    )

    stream = LLMChain(
        llm=llm,
        prompt=prompt
    )

    response = stream({"ques": ques})
    response_text = response.get('text', '')
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)