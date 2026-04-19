"""
create_agent를 사용한 도구 정의 (자동 실행)

- create_agent는 도구 호출 → 실행 → 결과 전달 → 재호출 루프를 자동으로 처리
- 01-tool.py에서 직접 구현했던 도구 호출 루프가 필요 없음
- args_schema로 도구 인자에 Pydantic BaseModel 검증을 추가 (01-tool.py와 동일)
- PydanticAI의 @agent.tool과 동일한 수준의 자동화

→ 01-tool.py와 비교하여 보일러플레이트가 얼마나 줄어드는지 확인

대응하는 pydantic 예제: pydantic/03-tool/01-tool.py
"""
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# --- 모델 정의 ---

class Search(BaseModel):
    keyword: str = Field(description="검색어", min_length=2)
    location: str = Field(description="위치", min_length=1)

# --- 도구 정의 (01-tool.py와 동일) ---

@tool(args_schema=Search)
def web_search(keyword: str, location: str) -> list[str]:
    """웹 검색을 해서 데이터를 가져옵니다."""
    return ["남산타워", "청와대", "글라스 하우스"]

@tool
def format_result(result: str) -> str:
    """결과를 포맷팅합니다."""
    return f"**-{result}-**"

# --- Agent 정의 (도구 실행 루프를 자동 처리) ---

agent = create_agent(
    "gpt-5.2",
    tools=[web_search, format_result],
    system_prompt="도시 정보를 정확히 알려줘. 결과는 100글자 이내로 작성해줘.",
)

# --- 실행 ---

def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "서울에 대해 알려줘. 서울에 있는 유명한 건물의 목록도 포함하고 결과를 포맷팅해줘."}]}
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
