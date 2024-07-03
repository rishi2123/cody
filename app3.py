import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai

def add_vertical_space(num_lines: int = 1):
    for _ in range(num_lines):
        st.write("")

# Custom CSS for styling
st.markdown("""
    <style>
        .chat-box {
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .input-area {
            display: flex;
            align-items: center;
        }
        .input-area input {
            flex: 1;
            margin-right: 10px;
        }
        .input-area button {
            flex: 0;
        }
        .icon-button {
            background: none;
            border: none;
            cursor: pointer;
        }
        .icon-button img {
            width: 24px;
            height: 24px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar contents
with st.sidebar:
    st.title('Cody 💬')
    st.markdown('''
    ## About
    Meet Cody, your PDF-savvy chatbot! Upload a PDF, ask Cody any question about it, and get instant, accurate answers. Cody uses advanced AI to understand and respond to your queries, making PDF interactions seamless and efficient.
    ''')
    add_vertical_space(5)

load_dotenv()

def main():
    st.header("Cody 💬")

    # Chat box
    chat_box = st.empty()

    # Input area
    query = st.text_input("Ask your question:", key="query")

    # Upload button next to the input field
    upload_button = st.file_uploader("", type='pdf', key="upload_button")

    # Store uploaded file globally
    if upload_button is not None:
        st.session_state.pdf = upload_button

    if query:
        if 'pdf' in st.session_state and st.session_state.pdf is not None:
            pdf_reader = PdfReader(st.session_state.pdf)

            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            store_name = st.session_state.pdf.name[:-4]

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings()
                embedding_vectors = embeddings.embed_documents(chunks)
                if embedding_vectors:
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                    with open(f"{store_name}.pkl", "wb") as f:
                        pickle.dump(VectorStore, f)
                else:
                    st.write("No embeddings generated")

            # Accept user questions/query
            if query:
                docs = VectorStore.similarity_search(query=query, k=3)

                llm = OpenAI(model_name='gpt-4')
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                chat_box.markdown(f"<div class='chat-box'>{response}</div>", unsafe_allow_html=True)
        else:
            llm = OpenAI(model_name='gpt-4')
            with get_openai_callback() as cb:
                response = llm(query)
                print(cb)
            chat_box.markdown(f"<div class='chat-box'>{response}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()