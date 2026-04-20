"""
구조화된 출력을 JSON으로 변환하는 방법

- 01-deps-and-output.py와 동일한 구조에서
  model_dump_json()을 사용해 출력을 JSON 문자열로 직렬화하는 예제
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
    print(response.output.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
