import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

os.environ['OPENAI_API_KEY'] = '<insert open ai key?'


def main():
    st.header('Chat with PDF')
    st.sidebar.title('LLM ChatApp using LangChain')
    st.sidebar.markdown("""
    This is an LLm powered chatbot built using LangChain and Streamlit""")
    pdf  =st.file_uploader("Upload PDF File here", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text = text + page.extract_text()

        #st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
        chunk_overlap=200,
        length_function = len)

        chunks = text_splitter.split_text(text=text)


        store_name = pdf.name[:-4]
        st.write(store_name)

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:        
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)

        query = st.text_input("Ask Question from you PDF file")

        if query:
            docs = VectorStore.similarity_search(query = query , k =3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type='stuff')

            response = chain.run(input_documents= docs, question = query)
            st.write(response)
            



if __name__ == "__main__":
    main()
