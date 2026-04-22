"""
AgentExecutor를 사용한 도구 정의 (레거시 방식)

- AgentExecutor는 LangGraph 도입 이전의 기존 에이전트 런타임
- create_tool_calling_agent로 에이전트를 정의한 뒤 AgentExecutor로 감싸 실행
- 프롬프트에 {agent_scratchpad} 슬롯을 직접 뚫어줘야 함
- 호출 시 {"input": "..."} 형식을 사용하고 결과는 result["output"]에 들어있음
- 01b-agent.py(create_agent)와 비교하면 설정 단계가 많고 저수준 API가 드러남

대응하는 pydantic 예제: pydantic/03-tool/01-tool.py
"""
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

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

# --- 프롬프트 정의 (agent_scratchpad 슬롯을 직접 추가) ---

# agent_scratchpad에는 AgentExecutor가 중간 도구 호출 기록을 채워 넣음
prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 100글자 이내로 작성해줘."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- Agent + Executor 정의 ---

llm = init_chat_model("gpt-4o", model_provider="openai")
tools = [web_search, format_result]

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

# --- 실행 ---

def main():
    result = executor.invoke(
        {"input": "서울에 대해 알려줘. 서울에 있는 유명한 건물의 목록도 포함하고 결과를 포맷팅해줘."}
    )
    print(result["output"])

if __name__ == "__main__":
    main()
