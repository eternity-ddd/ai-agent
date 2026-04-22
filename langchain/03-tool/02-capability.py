"""
LangChain 내장 도구 사용법

- OpenAI 네이티브 웹 검색: model_provider에서 제공하는 웹 검색 (PydanticAI의 WebSearch 대응)
- WebBaseLoader: 웹 페이지 내용을 가져오는 로더 (PydanticAI의 WebFetch 대응)
- PydanticAI는 capabilities=[WebSearch(), WebFetch()] 한 줄이지만,
  LangChain은 각 도구를 개별 생성하고, 도메인 제한은 직접 구현해야 함

대응하는 pydantic 예제: pydantic/03-tool/02-capability.py
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# --- 도구 정의 ---

# PydanticAI의 WebFetch(allowed_domains=['aladin.co.kr']) 대응
@tool
def web_fetch(url: str) -> str:
    """URL의 웹 페이지 내용을 가져옵니다. aladin.co.kr 도메인만 허용됩니다."""
    print("알라딘 페이지 읽는 중...")
    if "aladin.co.kr" not in url:
        return "허용되지 않는 도메인입니다. aladin.co.kr만 접근 가능합니다."
    docs = WebBaseLoader(url).load()
    text = "\n".join(doc.page_content for doc in docs)
    return text[:10000]

# --- Agent 정의 ---

# PydanticAI의 WebSearch() 대응: OpenAI 네이티브 웹 검색을 사용
# tools에 {"type": "web_search_preview"} 도구 스펙을 전달하면 LangChain이
# 내부적으로 OpenAI Responses API로 라우팅해 네이티브 웹 검색이 활성화됨
agent = create_agent(
    init_chat_model("gpt-4o", model_provider="openai"),
    tools=[{"type": "web_search_preview"}, web_fetch],
)

# --- 실행 ---

def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 삼성전자 주가를 조회해주고, aladin.co.kr에서 프로젝트 헤일메리 책 페이지를 찾아서 책 소개 부분을 요약해줘"}]}
    )
    # OpenAI Responses API의 네이티브 도구 사용 시 content가 리스트로 반환됨
    content = result["messages"][-1].content
    if isinstance(content, list):
        print(content[0]["text"])
    else:
        print(content)

if __name__ == "__main__":
    main()
