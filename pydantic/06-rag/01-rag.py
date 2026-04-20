"""
PydanticAI로 RAG 구현 (PDF → ChromaDB)

PydanticAI에는 벡터 DB, 임베딩, 문서 로더 등 RAG 관련 기능이 내장되어 있지 않아
모든 파이프라인을 직접 구현해야 한다:
- PyMuPDF로 PDF 텍스트 추출을 직접 구현
- 텍스트 청킹 로직을 직접 구현
- OpenAI API로 임베딩 직접 호출
- ChromaDB 클라이언트를 직접 생성/관리
- 검색 로직을 @agent.tool로 등록

→ LangChain 버전(langchain/06-rag/01-rag.py)과 비교하면 보일러플레이트가 많음
"""
import asyncio
import os
from dataclasses import dataclass

import fitz  # PyMuPDF
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic_ai import Agent, RunContext

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "../../data/jeju_guide.pdf")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "./chroma_db")

# --- PDF 텍스트 추출을 직접 구현해야 함 ---

def extract_text_from_pdf(path: str) -> str:
    """PyMuPDF로 PDF에서 텍스트 추출"""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# --- 텍스트 청킹도 직접 구현해야 함 ---

def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """텍스트를 chunk_size 단위로 분할 (overlap 포함)"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

# --- OpenAI 임베딩도 직접 호출해야 함 ---

openai_client = OpenAI()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """OpenAI API로 임베딩을 직접 호출"""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]

# --- PDF → 청킹 → 임베딩 → ChromaDB 저장 ---

print("PDF 텍스트 추출 중...")
raw_text = extract_text_from_pdf(PDF_PATH)
chunks = split_text(raw_text)
print(f"청킹 완료: {len(chunks)}개 청크")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="jeju_guide")

if collection.count() == 0:
    print("임베딩 중...")
    embeddings = embed_texts(chunks)
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    print(f"임베딩 완료: {len(chunks)}건 저장")
else:
    print(f"기존 데이터 사용: {collection.count()}건")

# --- 검색 함수도 직접 구현해야 함 ---

# --- 의존성 ---

@dataclass
class RagDeps:
    collection: chromadb.Collection

# --- Agent 정의 ---

agent = Agent[RagDeps](
    "openai:gpt-4o",
    instructions="검색된 문서를 기반으로 정확하게 답변해줘. 문서에 없는 내용은 추측하지 마.",
    deps_type=RagDeps,  # type: ignore
)

@agent.tool
def retrieve(ctx: RunContext[RagDeps], query: str) -> str:
    """질문과 관련된 문서를 검색합니다.

    Args:
        query: 검색할 질문
    """
    query_embedding = embed_texts([query])
    results = ctx.deps.collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )
    return "\n\n".join(results["documents"][0])

# --- 실행 ---

async def main():
    deps = RagDeps(collection=collection)

    response = await agent.run(
        user_prompt="제주도의 해녀 문화에 대해 알려줘",
        deps=deps,
    )
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
