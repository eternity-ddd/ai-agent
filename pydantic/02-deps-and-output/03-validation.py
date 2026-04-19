"""
Pydantic field_validator를 활용한 입력/출력 검증

- deps(MyState)에 field_validator를 적용해 입력값을 검증하는 방법
- output_type(CityInfo)에 field_validator를 적용해 LLM 출력값을 검증하는 방법
- retries=0으로 설정하여 검증 실패 시 재시도 없이 즉시 에러를 발생시킴
- 아래 예제는 length=1000000을 전달하므로 validator에 의해 즉시 ValidationError 발생 (의도된 실패 예제)
"""
import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from pydantic_ai import Agent, RunContext

load_dotenv()

# --- 모델 정의 (validator 포함) ---

class MyState(BaseModel):
    length: int

    @field_validator('length')
    @classmethod
    def check_length(cls, v: int) -> int:
        if v >= 10000:
            raise ValueError("길이는 10000보다 작아야 한다.")
        return v

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

    @field_validator('population')
    @classmethod
    def check_population(cls, v: int) -> int:
        if v >= 100000:
            raise ValueError("인구는 100000보다 작아야 한다.")
        return v

# --- Agent 정의 ---

agent = Agent[MyState, CityInfo](
    "openai:gpt-5.2",
    instructions="도시 정보를 정확히 알려줘",
    deps_type=MyState,      # type: ignore
    output_type=CityInfo,   # type: ignore
    retries=0,
)

@agent.instructions
def length(ctx: RunContext[MyState]):
    return f"결과는 {ctx.deps.length} 글자 이내로 작성해줘."

# --- 실행 (의도된 실패: length=1000000은 validator에 의해 거부됨) ---

async def main():
    response = await agent.run(user_prompt="서울에 대해 알려줘", deps=MyState(length=1000000))
    print(response.output.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
