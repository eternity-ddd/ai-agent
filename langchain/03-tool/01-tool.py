"""
@tool 데코레이터와 bind_tools()를 사용한 도구 정의 (수동 실행)

- @tool: 함수를 LLM이 호출할 수 있는 도구로 변환
- args_schema로 도구 인자에 Pydantic BaseModel 검증을 추가
- bind_tools()는 LLM에게 도구 스키마만 전달할 뿐, 실제 실행은 하지 않음
- 개발자가 tool_calls를 꺼내서 직접 실행하는 루프를 구현해야 함

→ create_agent를 사용하면 이 루프가 자동 처리됨 (01a-agent.py 참고)

대응하는 pydantic 예제: pydantic/03-tool/01-tool.py
"""
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv()

# --- 모델 정의 ---

class Search(BaseModel):
    keyword: str = Field(description="검색어", min_length=2)
    location: str = Field(description="위치", min_length=1)

# --- 도구 정의 ---

@tool(args_schema=Search)
def web_search(keyword: str, location: str) -> list[str]:
    """웹 검색을 해서 데이터를 가져옵니다."""
    return ["남산타워", "청와대", "글라스 하우스"]

@tool
def format_result(result: str) -> str:
    """결과를 포맷팅합니다."""
    return f"**-{result}-**"

# --- 체인 정의 (프롬프트 | LLM + 도구 바인딩) ---

tools = [web_search, format_result]
tools_map = {t.name: t for t in tools}

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 100글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm_with_tools = init_chat_model("gpt-5.2", model_provider="openai").bind_tools(tools)
chain = prompt | llm_with_tools

# --- 실행 ---

def main():
    response = chain.invoke(
        {"question": "서울에 대해 알려줘. 서울에 있는 유명한 건물의 목록도 포함하고 결과를 포맷팅해줘."}
    )

    messages = [response]
    for tool_call in response.tool_calls:
        result = tools_map[tool_call["name"]].invoke(tool_call["args"])
        messages.append({"role": "tool", "content": str(result), "tool_call_id": tool_call["id"]})

    # 재호출 시에는 prompt를 거치지 않고 llm만 사용 (messages 리스트를 직접 전달)
    final_response = llm_with_tools.invoke(messages)
    print(final_response.content)

if __name__ == "__main__":
    main()
