from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
import os

os.environ['OPENAI_API_KEY'] = '<insert open ai key?'

def main():
     st.header('Youtube Video Summarize')
     query = st.text_input("Input youtube url link")

     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

     if query:
          loader = YoutubeLoader.from_youtube_url(query, add_video_info=True)
          result = loader.load()
          llm = OpenAI(temperature=0.0)
          text = text_splitter.split_documents(result)  
          chain = load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
        
          response = chain.run(text)
          st.write(response)






if __name__ == "__main__":
    main()