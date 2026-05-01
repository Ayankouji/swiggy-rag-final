import streamlit as st
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

st.set_page_config(page_title="Swiggy RAG Chat", layout="wide")
st.title("Swiggy Annual Report Chat")
st.write("Ask anything about Swiggy FY 2023-2024 report")


@st.cache_resource
def load_rag():
    docs = PyPDFLoader("Annual-Report-FY-2023-24.pdf").load()

    for i, doc in enumerate(docs):
        doc.metadata["page"] = i

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15}
    )

    return retriever

retriever = load_rag()


llm = ChatGroq(model_name="llama-3.1-8b-instant")


def rewrite_query(query):
    prompt = f"""
Convert this into a precise financial query.
Fix spelling mistakes and expand abbreviations.

Query: {query}
"""
    return llm.invoke(prompt).content.strip()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    improved_query = rewrite_query(query)

    docs = retriever.invoke(improved_query)

    context = "\n\n".join([doc.page_content for doc in docs])

 
    prompt = f"""
You are an AI assistant answering questions from a financial report.

Rules:
- Use ONLY the provided context
- If answer is directly present → return it clearly
- If answer is not directly stated → infer from context (but do NOT guess)
- For reasoning questions → summarize relevant points
- If a concept is not mentioned → clearly say it is not explicitly mentioned
- Always answer in full sentence with proper explanation
- Do NOT hallucinate

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("📄 Source Pages"):
        pages = sorted(set([doc.metadata.get("page") for doc in docs]))
        st.write("Pages:", pages)

        for doc in docs:
            st.markdown(f"**Page {doc.metadata.get('page')}**")
            st.write(doc.page_content[:250])
            st.write("---")