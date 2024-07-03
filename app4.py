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
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f5f5f5; /* Light gray background */
            color: #000; /* Black text for readability */
        }
        .chat-box .user {
            font-weight: bold;
            color: #007bff; /* Streamlit blue for user */
        }
        .chat-box .bot {
            font-weight: normal;
            color: #333; /* Darker gray for bot */
        }
        .input-area {
            display: flex;
            align-items: center;
            margin-top: 10px;
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
    st.title('💬Chat App')
    st.markdown('''
    ## About
    #####Chat App can make mistakes. Check important info.
    ''')
    add_vertical_space(5)

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'pdf' not in st.session_state:
        st.session_state.pdf = None

    # Display chat history
    chat_box = st.empty()
    for chat in st.session_state.chat_history:
        chat_box.markdown(f"<div class='chat-box'><div class='user'>User:</div> {chat['query']}<br><div class='bot'>Bot:</div> {chat['response']}</div>", unsafe_allow_html=True)

    # Input area
    with st.form(key='input_form', clear_on_submit=True):
        query = st.text_input("Ask your question:", key="query")
        upload_button = st.file_uploader("", type='pdf', key="upload_button")
        submit_button = st.form_submit_button(label='Submit')

    if submit_button and query:
        # Store uploaded file globally
        if upload_button is not None:
            st.session_state.pdf = upload_button

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
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name='gpt-4')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
        else:
            llm = OpenAI(model_name='gpt-4')
            with get_openai_callback() as cb:
                response = llm(query)
                print(cb)

        # Update chat history
        st.session_state.chat_history.append({"query": query, "response": response})

        # Refresh chat box with updated history
        chat_box.empty()
        for chat in st.session_state.chat_history:
            chat_box.markdown(f"<div class='chat-box'><div class='user'>User:</div> {chat['query']}<br><div class='bot'>Bot:</div> {chat['response']}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
