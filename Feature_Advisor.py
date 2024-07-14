import streamlit as st
import os

from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain import ConversationChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tiktoken
import yaml

from utils import load_settings, load_inputs, split_documents, load_db, generate_response

st.set_page_config(page_title="Feature Advisor", page_icon=":material/smart_toy:")

settings = load_settings()
cate = settings['cate']
query_template_product_simple = settings['prompts']['product_simple']
query_template_profile = settings['prompts']['profile']

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if 'openai_api_key' in st.session_state:
    openai_api_key = st.session_state.openai_api_key
else:
    openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

if openai_api_key != "" and 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = openai_api_key


genre = st.sidebar.radio(
    "질문의 유형을 선택하세요",
    ["상품 특성 기반 추천", "프로파일 기반 추천"],
    captions = ["상품에 대한 특성을 기반으로 추천합니다.","프로파일을 기반으로 추천합니다." ])


# app
st.title('🤖Feature Advisor')

with st.form('my_form'):
    if "상품 특성 기반 추천" in genre:
        text = st.text_area('Enter Text:', 
                            label_visibility=st.session_state.visibility,
                            disabled=st.session_state.disabled,
                            placeholder='타겟팅하고자 하는 상품의 특성을 입력 (특성, 소구점, 타겟고객 등)'
                            )
    else:
        text = st.text_area('Enter Text:', 
                            label_visibility=st.session_state.visibility,
                            disabled=st.session_state.disabled,
                            placeholder= '프로파일을 입력해주세요 (고객, 상품, 정서, 상황 등)'
                            )
    submitted = st.form_submit_button('Submit')

    
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text, "./db/faiss_index_1", genre, openai_api_key)


