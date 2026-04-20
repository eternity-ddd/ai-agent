"""
deps_type(상태)과 output_type(구조화된 출력) 사용법

- deps_type: 에이전트 실행 시 외부에서 주입하는 의존성/상태 (dataclass 사용)
- output_type: 에이전트의 응답을 Pydantic BaseModel로 구조화
- RunContext를 통해 instructions에서 deps에 접근하는 방법
"""
import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

load_dotenv()

# --- 모델 정의 ---

@dataclass
class MyState:
    length: int

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

# --- Agent 정의 ---

agent = Agent[MyState, CityInfo](
    "openai:gpt-4o",
    instructions="도시 정보를 정확히 알려줘",
    deps_type=MyState,      # type: ignore
    output_type=CityInfo,   # type: ignore
)

@agent.instructions
def length(ctx: RunContext[MyState]):
    return f"결과는 {ctx.deps.length} 글자 이내로 작성해줘."

# --- 실행 ---

async def main():
    response = await agent.run(user_prompt="서울에 대해 알려줘", deps=MyState(200))
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
