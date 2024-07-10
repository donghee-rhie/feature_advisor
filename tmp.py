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

openai_api_key = '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'
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
# embeddings = embeddings_model.embed_documents([doc.page_content for doc in texts])


# Load into Vector Database
db = FAISS.from_documents(texts, embeddings_model)
db.save_local("./db/faiss_index_1")