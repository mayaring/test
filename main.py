from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import tempfile
import os

st.title("ChatPDF")
st.write('---')

uploaded_file = st.file_uploader("PDF 파일을 올려주세요!",type=['pdf'])
st.write('---')

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages=pdf_to_document(uploaded_file)

    # SPLIT
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    # EMBEDDIG
    embeddings_model = OpenAIEmbeddings()

    # LOAD IT INTO CHROMA
    db = chroma.Chroma.from_documents(texts,embeddings_model)


    # QUSTION
    st.header('PDF에서 질문해보세요!')
    question = st.text_input('질문을 입력하세요')

    if st.button("질문하기"):
        with st.spinner('진행 중...'):
            llm = ChatOpenAI(temperature=0)
            qa_chain= RetrievalQA.from_chain_type( llm,retriever=db.as_retriever())

            result=qa_chain({"query":question} )
            st.write(result["result"])
