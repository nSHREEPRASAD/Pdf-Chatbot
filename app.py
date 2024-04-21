# IMPORT LIBRARIES

import streamlit as st
import time
from streamlit import config
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space

# TEXT-EXTRACTION FUNCTION

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
# GET TEXT-CHUNKS FUNCTION

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

# SIDEBAR

with st.sidebar:

    # UPLOAD PDF FILES

    pdfs=st.file_uploader(":green[*Upload a PDF 📄*]",accept_multiple_files=True,key="only my pdfs", type="pdf")

    add_vertical_space(1)

    st.title('💬🗨️ LLM Chat-App')

    st.markdown('''
    ##About
    This App is an LLM-PDF-ChatBot built using:
                
    -[Streamlit](https://streamlit.io/)
                
    -[LangChain](https://www.langchain.com/)
                
    -[OpenAI](https://openai.com/)
                
     ''')

def main():

    # PDF_SIZE LIMIT

    config.set_option('server.maxUploadSize', 1000 * 1024 )

    st.header(":green[*Chat with your PDF 🗨️📄*]")

    add_vertical_space(1)

    # CHAT-HISTORY TO DISPLAY

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if pdfs:

        # GET TEXT
        
        text = get_pdf_text(pdfs)

        # GET CHUNKS
        
        text_chunks = get_text_chunks(text)

        # API_KEY

        load_dotenv()
        openai_api_key = os.environ["OPENAI_API_KEY"]

        # EMBEDDINGS

        embeddings=OpenAIEmbeddings(api_key=openai_api_key)

        # CHROMADB VECTORSTORE

        db=chroma.Chroma.from_texts(text_chunks,embeddings)

        # RETRIEVER

        retriever = db.as_retriever(search_type="similarity")
        qa= ConversationalRetrievalChain.from_llm(OpenAI(api_key=openai_api_key),retriever)
        
        # CHAT-HISTORY FOR CONVERSATION MEMORY

        chat_history = st.session_state.get("chat_history", [])


    # EXTRACTING ANSWER
        
    if query := st.chat_input("Enter Prompt :"):
        
        # ADDING QUESTION IN CHAT-HISTORY TO DISPLAY

        st.session_state.messages.append({"role": "user", "content": query})

        # DISPLAYING QUESTION

        with st.chat_message("user"):
            st.markdown(query)

        # DISPLAYING ANSWER
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # GENERATING ANSWER USING ConversationalRetrievalChain

            result = qa({"question":query, "chat_history":chat_history})
            time.sleep(25)

            message_placeholder.markdown(result["answer"])

        # ADDING ANSWER IN CHAT-HISTORY TO DISPLAY
            
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

        # ADDING ANSWER IN CHAT-HISTORY FOR MEMORY

        chat_history.append((query,result["answer"]))
        st.session_state.chat_history=chat_history

        

if __name__ == '__main__':
    main()
    