import streamlit as st

import os
import time
from datetime import datetime

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
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken
import yaml

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # "/" is a marker to show difference 
        # you don't need it 
        self.text+=token
        self.container.markdown(self.text) 

def get_sorted_files_by_creation_time(directory):
    files_with_times = [(file, os.path.getctime(os.path.join(directory, file))) for file 
                            in os.listdir(directory) 
                            if os.path.isfile(os.path.join(directory, file))]
    sorted_files = sorted(files_with_times, key=lambda x: x[1], reverse=True)
    return [file for file, ctime in sorted_files]

def load_settings():
    with open('settings.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data

@st.cache_data
def load_inputs():
    path = './rag_input/category'
    loader = DirectoryLoader(path=path)
    data = loader.load()
    print("FINISHED : LOAD INPUTS")
    return data


def split_documents(data):
    tokenizer = tiktoken.get_encoding("cl100k_base")

    def tiktoken_len(text):
        tokens = tokenizer.encode(text)
        return len(tokens)

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, length_function = tiktoken_len
    )   
    texts = text_splitter.split_documents(data)
    print("FINISHED : SPLIT TEXT")
    return texts


def build_and_save_db(texts, embedding_model, openai_api_key, name):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings_model)
    db.save_local("./db/{}".format(name))


def load_db(path, openai_api_key):
    
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local(path, 
                            embeddings_model, 
                            allow_dangerous_deserialization=True
                            )

def generate_response(input_text, path_db, genre, openai_api_key):

    settings = load_settings()

    cate = settings['cate']
    query_template_product = settings['prompts']['product']
    query_template_profile = settings['prompts']['profile']
    query_template_product_simple = settings['prompts']['product_simple']

    # Load vector Database
    db = load_db(path_db, openai_api_key)

    chat_box=st.empty() 
    stream_handler = StreamHandler(chat_box)

    # Retriever
    openai = ChatOpenAI(model_name="gpt-4o-2024-05-13",
                        streaming=True, 
                        callbacks=[stream_handler],
                        openai_api_key=openai_api_key,
                        temperature = 0)

    qa = RetrievalQA.from_chain_type(llm = openai,
                                    chain_type = "stuff", # stuff, map_reduce, refine, map_rerank
                                    retriever = db.as_retriever(
                                        search_type="mmr",
                                        search_kwargs={'k':3, 'fetch_k': 10}),
                                    return_source_documents = True)

    if "상품 특성 기반 추천" in genre:
        query = query_template_product_simple.format(description=input_text, cate=cate)
    else:
        query = query_template_profile.format(description=input_text, cate=cate)

    answer = qa(query)

    if genre == "상품 특성 기반 추천":
        summary = answer['result'].split("\n")[2].replace("\"","")
        profiling = answer['result'].split("카테고리 추천")[0][:-10]
        recommendation = answer['result'].split("카테고리 추천")[1]
    else:
        summary = answer['result'].split("\n")[1].replace("\"","")
        profiling = answer['result'].split("카테고리 추천")[0][:-10]
        recommendation = answer['result'].split("카테고리 추천")[1]    

    # make additional page
    file_content = '''
import streamlit as st
st.set_page_config(layout='wide')

st.title("{summary}")

# Displaying the value of A on this new page
st.markdown("""{input}""")

col1, col2 = st.columns(2, gap='large')

with col1:
    st.header("고객 특성")
    st.markdown("""{profiling}""")

with col2:
    st.header("카테고리 추천")
    st.markdown("""{recommendation}""")
'''.format(summary=summary, 
           input=input_text, 
           profiling=profiling, 
           recommendation=recommendation, 
           answer=answer['result'])
    
    with open("donghee-rhie/feature_advisor/main/pages/{}.py".format(summary), 'w') as f:
        f.write(file_content)



# embeddings = embeddings_model.embed_documents([doc.page_content for doc in texts])
# Load into Vector Database
# db = FAISS.from_documents(texts, embeddings_model)
# db.save_local("./db/faiss_index_1")
