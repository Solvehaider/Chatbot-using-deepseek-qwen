import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import base64
import gc
import uuid

from IPython.display import Markdown, display
import os
from langchain_core.output_parsers import StrOutputParser

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader

import streamlit as st

# login into aliababa cloud and from the model playgroung create api for yourself
qwen_api = "your alibaba api"

pdfs_directory = "directory path where you can save pdf"

embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
# model = ChatGroq(model="deepseek-r1-distill-llama-70b",api_key=GROQ_API_KEY)
# model = OllamaLLM(
#     model="deepseek-r1:1.5b", verbose=True, mirostat_eta=0.9, mirostat_tau=2
# )
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

messages = [
    {
        "role": "system",
        "content": """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.""",
    }
]



client = None


@st.cache_resource
def load_llm(model_option):
    if model_option == "Qwen-Max":
        llm = OpenAI(
            # If the environment variable is not configured, replace the following line with: api_key="sk-xxx",
            api_key=qwen_api,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        return llm
    elif model_option == "DeepSeek-R1":
        llm = OllamaLLM(
            model="deepseek-r1:1.5b", verbose=True, mirostat_eta=0.9, mirostat_tau=2
        )
        return llm


def get_response(messages, model, user_query, relevant_info):

    messages.append({"role": "user", "content": user_query})
    messages.append({"role": "assistant", "content": f"Relevant info: {relevant_info}"})

    completion = model.chat.completions.create(model="qwen-plus", messages=messages)

    return completion


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents


def index_docs(documents):
    vector_store.add_documents(documents)


def retrieve_docs(query):
    return vector_store.similarity_search(query)


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    return text_splitter.split_documents(documents)


with st.sidebar:
    # Add dropdown for model selection
    model_option = st.selectbox(
        "Select Model",
        ("Qwen-Max", "DeepSeek-R1"),
        index=None,
        placeholder="Select Model",
    )
    model = load_llm(model_option)
    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            if upload_pdf(uploaded_file):
                st.success("File uploaded successfully.")
            documents = load_pdf(pdfs_directory + uploaded_file.name)
            chunked_documents = split_text(documents)
            index_docs(chunked_documents)

            st.success("Ready to Chat!")
            display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns(
    [6, 1],
)

with col1:
    st.header(f"DeepSeek-R1 And Qwen Chatbot")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    related_documents = retrieve_docs(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        fullres = ""
        context = "\n\n".join([doc.page_content for doc in related_documents])
        prompt2 = ChatPromptTemplate.from_template(template)
        parser = StrOutputParser()
        if model_option == "DeepSeek-R1":
            chain = prompt2 | model | parser
            fullres = chain.invoke({"question": prompt, "context": context})
            # st.write("Using Qwen-Max model...")
        elif model_option == "Qwen-Max":
            # st.write("Using DeepSeek-R1 model...")
            fullres = (
                get_response(messages, model, prompt, context)
                .choices[0]
                .message.content
            )
            messages.append({"role": "assistant", "content": fullres})

            # fullres = get_response(prompt2, model)

        # fullres = ""
        # for chunk in chain.stream({"question": prompt, "context": context}):
        #     fullres += chunk
        #     message_placeholder.markdown(fullres + "▌")

        message_placeholder.markdown(fullres)
        st.session_state.messages.append({"role": "assistant", "content": fullres})
