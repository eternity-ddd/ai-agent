"""
LangChain으로 RAG 구현 (PDF → ChromaDB)

LangChain은 RAG에 필요한 모든 구성 요소가 내장되어 있어
파이프(|)로 깔끔하게 연결할 수 있다:
- PyMuPDFLoader: PDF 로드 (1줄)
- RecursiveCharacterTextSplitter: 텍스트 청킹 (1줄)
- Chroma.from_documents(): 임베딩 + 벡터 DB 저장 (1줄)
- 파이프(|)로 검색 → 프롬프트 → LLM → 출력 체인 연결

→ PydanticAI 버전과 비교하면 보일러플레이트가 적음

대응하는 pydantic 예제: pydantic/06-rag/01-rag.py
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "../../data/jeju_guide.pdf")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "./chroma_db")

# --- PDF → 청킹 → 벡터 DB (각 1줄) ---

docs = PyMuPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=CHROMA_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"PDF 로드 완료: {len(docs)}페이지, {len(chunks)}개 청크")

# --- RAG 체인 정의 (검색 | 프롬프트 | LLM | 출력파서) ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "검색된 문서를 기반으로 정확하게 답변해줘. 문서에 없는 내용은 추측하지 마.\n\n{context}"),
    ("human", "{question}"),
])

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | init_chat_model("gpt-5.2", model_provider="openai")
    | StrOutputParser()
)

# --- 실행 ---

def main():
    response = chain.invoke("제주도의 해녀 문화에 대해 알려줘")
    print(response)

if __name__ == "__main__":
    main()
