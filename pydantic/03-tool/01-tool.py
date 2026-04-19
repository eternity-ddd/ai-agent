"""
@agent.tool과 @agent.tool_plain을 사용한 도구 정의

- @agent.tool: RunContext를 첫 번째 인자로 받아 deps(상태)에 접근 가능
- @agent.tool_plain: RunContext 없이 순수 함수로 동작하는 도구
- 도구 인자를 Pydantic BaseModel(Search)로 구조화하여 Field의 description, min_length 등 검증 추가
- 에이전트가 프롬프트를 분석하여 적절한 도구를 자동으로 선택하고 호출함
"""
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()

# --- 모델 정의 ---

class MyState(BaseModel):
    length: int
    building: str

class Search(BaseModel):
    keyword: str = Field(description="검색어", min_length=2)
    location: str = Field(description="위치", min_length=1)

# --- Agent 정의 ---

agent = Agent[MyState](
    "openai:gpt-5.2",
    instructions="도시 정보를 정확히 알려줘",
    deps_type=MyState,  # type: ignore
)

@agent.instructions
def length(ctx: RunContext[MyState]):
    return f"결과는 {ctx.deps.length} 글자 이내로 작성해줘."

# --- 도구 정의 ---

@agent.tool
def web_search(ctx: RunContext[MyState], search: Search) -> list[str]:
    """웹 검색을 해서 데이터를 가져옵니다."""
    return ["남산타워", "청와대", ctx.deps.building]

@agent.tool_plain
def format_result(result: str) -> str:
    """결과를 포맷팅합니다."""
    return f"**-{result}-**"

# --- 실행 ---

async def main():
    response = await agent.run(
        user_prompt="서울에 대해 알려줘. 서울에 있는 유명한 건물의 목록도 포함하고 결과를 포맷팅해줘.",
        deps=MyState(length=100, building="글라스 하우스"),
    )
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
