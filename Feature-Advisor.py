import streamlit as st

from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import tiktoken

cate = [
  'TV',
'가방잡화',
'결혼',
'경력개발',
'고가상품',
'고화질영상',
'공기질',
'국내여행',
'낚시',
'남성패션',
'농구',
'다이어트',
'달리기',
'독서',
'등산',
'렌탈',
'반려동물',
'보석_명품',
'보험',
'사진_카메라',
'숙박업체',
'스포츠운동용품',
'신발',
'아동패션',
'어학공부',
'여성패션',
'온라인상품권',
'요가',
'요리',
'유아교육',
'의료_제약',
'인테리어',
'자전거',
'주식투자',
'중고등교육',
'지도네비게이션',
'초등교육',
'초등학교저학년',
'축구',
'출산',
'투자정보',
'필라테스',
'해외여행',
'향수화장품',
'헬스',
'홈케어서비스',
'단말기파손',
'안전_방역',
'영화',
'멤버십포인트',
'편의점',
'복권',
'이사',
'음악스트리밍',
'아이돌',
'렌트',
'팟캐스트_라디오',
'VR',
'성인콘텐츠',
'과정외교육',
'프로야구',
'뮤머',
'반려동물관심',
'반려동물보유',
'국책학생(유학생)',
'대학생',
'가정외교육',
'부모서비스',
'의료_제약',
'유기농제품',
'구매력',
'은행',
'의사결정자',
'카드',
'저신용자',
'핀테크',
'간편결제',
'세금',
'가상화폐',
'전기요금',
'패스트패션브랜드',
'명품의류',
'패스트푸드',
'배달음식',
'이사서비스',
'부동산중개',
'안전_방범',
'홈케어서비스',
'동영상스트리밍',
'유튜브',
'저가선호',
'E-BOOK',
'AR콘텐츠,'
'대형마트',
'구매전탐색',
'오픈마켓',
'홈쇼핑',
'해외직접구매',
'백화점_면세점',
'가격비교',
'주차서비스',
'대중교통',
'모빌리티공유서비스',
'블로그',
'온라인카페',
'인스타그램',
'SNS',
'페이스북',
'온라인뉴스',
'날씨',
'고객센터_서비스센터',
'모바일상품',
'홈상품',
'MVNO',
'삼성모바일',
'애플모바일',
'5G',
'속도측정',
'캐시백',
'유흥_도박',
'공공기관'
]

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

genre = st.sidebar.radio(
    "질문의 유형을 선택하세요",
    ["상품 특성 기반 추천", "프로파일 기반 추천"],
    captions = ["상품에 대한 특성을 기반으로 추천합니다.","프로파일을 기반으로 추천합니다." ])

print(genre)

# load inputs

path = './rag_input/category'
loader = DirectoryLoader(path=path)
data = loader.load()


# split documents

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0, length_function = tiktoken_len
)
texts = text_splitter.split_documents(data)


# Text Embedding

embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
# embeddings = embeddings_model.embed_documents([doc.page_content for doc in texts])


# Load into Vector Database
# db = FAISS.from_documents(texts, embeddings_model)
# db.save_local("./db/faiss_index_1")

# Load vector Database
db = FAISS.load_local("./db/faiss_index_1", embeddings_model, allow_dangerous_deserialization=True)

# Retriever
openai = ChatOpenAI(model_name="gpt-4o-2024-05-13",
                    streaming=True, 
                    # callbacks=[StreamingStdOutCallbackHandler()],
                    openai_api_key=openai_api_key,
                    temperature = 0)


qa = RetrievalQA.from_chain_type(llm = openai,
                                 chain_type = "stuff",
                                 retriever = db.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={'k':3, 'fetch_k': 10}),
                                 return_source_documents = True)


st.title('Feature Advisor')

query_template_product = '''
너는 마케팅 기획자야.
아래 조건을 반영해서 주요 소구 고객을 찾기 위한 카테고리를 추천하는 것이 목표야

A.상품의 특성은 아래와 같아

{description}

B. 해당 회원군에 대한 전반적인 설명을 해줘
 - 어떤 데모그래픽을 갖고 있는지, 주된 심리적 동인은 무엇인지, 어떤 것을 소비하는데 관심이 많은지 알려줘 
 - 이들은 주로 어떤 생각을 하고 있으며, 어떤 가치를 중요하게 생각하고, 궁극적으로 무엇을 원하는지에 대해 알려줘
 - 1.데모, 2.주요 심리적 동인, 3.주요 가치, 4. 주요 관심사의 항목으로 나눠서 작성해줘

C.이에 어떤 카테고리를 참조하여 고객을 찾아야 하는지, 왜 해당 카테고리를 추천하는지 알려줘
 - 카테고리는 {cate} 안의 카테고리 중에서 추천해줘
 - 먼저 제공된 문서만 기반으로 해서 카테고리 5개 추천, 그리고 그 이유를 설명해줘. 항목의 제목은 '주요 카테고리 추천' 으로 해줘.
 - 그리고 나서 제공된 문서와 네 생각을 모두 포함해서 카테고리 5개를 추천, 그리고 이유를 설명해줘. '주요 카테고리 추천'에서 추천한 항목은 또 추천하지 않아도 괜찮아. 항목의 제목은 별도로 만들지 말고, '주요 카테고리 추천'의 하위 항목으로 넣어줘
 - 추천 카테고리는 별도의 하위항목 없이 1~10으로 추가해줘
 

모든 답변은 자세히 풀어서 설명해줘
A-B-C의 순서대로 답변을 작성해줘
'''


query_template_profile = '''
너는 인간의 심리와 소비에 대해 연구하는 학자야.
특정 묘사를 기반으로 그것과 가장 어울리는 카테고리를 추천하는 것이 목표야.

A.묘사의 내용은 아래와 같아

{description}

B. 묘사된 대한 전반적인 설명을 해줘
 - 해당 묘사의 주된 심리적 동인은 무엇인지 설명해줘
 - 어떤 생각에 기반하고 있고, 어떤 가치관에 가까우며, 궁극적으로 무엇을 원하는지에 대해 알려줘
 - 해당 묘사를 간단하게 요약하고, 이 묘사에 부합하는 사람들의 특성에 대해서 설명해줘
 - 1.요약 프로파일, 2.주요 심리적 동인, 3.주요 가치, 4. 주요 관심사의 항목으로 나눠서 작성해줘

C.이에 어떤 카테고리를 참조하여 고객을 찾아야 하는지, 왜 해당 카테고리를 추천하는지 알려줘
 - 카테고리는 {cate} 안의 카테고리 중에서 추천해줘
 - 먼저 제공된 문서만 기반으로 해서 카테고리 5개 추천, 그리고 그 이유를 설명해줘. 항목의 제목은 '주요 카테고리 추천' 으로 해줘.
 - 그리고 나서 제공된 문서와 네 생각을 모두 포함해서 카테고리 5개를 추천, 그리고 이유를 설명해줘. '주요 카테고리 추천'에서 추천한 항목은 또 추천하지 않아도 괜찮아. 항목의 제목은 별도로 만들지 말고, '주요 카테고리 추천'의 하위 항목으로 넣어줘
 - 추천 카테고리는 별도의 하위항목 없이 1~10으로 추가해줘
 
모든 답변은 자세히 풀어서 설명해줘
A-B-C의 순서대로 답변을 작성해줘
'''


def generate_response(input_text):
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    
    if genre == "상품 특성 기반 추천":
        query = query_template_product.format(description=input_text, cate=cate)
    else:
        query = query_template_profile.format(description=input_text, cate=cate)
    
    answer = qa(query)
    st.markdown(answer['result'])

with st.form('my_form'):
    if genre == "상품 특성 기반 추천":
        text = st.text_area('Enter text:', '타겟팅하고자 하는 상품의 특성을 입력 (특성, 소구점, 타겟고객 등)')
    else:
        text = st.text_area('Enter text:', '프로파일을 입력해주세요 (고객, 상품, 정서, 상황 등)')

    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)