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

# Usage
add_vertical_space(3)  # Adds 3 lines of vertical space
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    Upload your PDF and get answers for your queries!
    ''')
    add_vertical_space(5)

load_dotenv()

def main():
    st.header("Chat with PDF 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

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

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

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
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(model_name='gpt-4')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    else:
        st.write("No PDF uploaded. You can ask general questions.")
        
        # Accept general user questions/query
        query = st.text_input("Ask your question:")

        if query:
            llm = OpenAI(model_name='gpt-4')
            with get_openai_callback() as cb:
                response = llm(query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
