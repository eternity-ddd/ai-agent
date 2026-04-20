"""
Agent 기본 사용법과 @agent.instructions 데코레이터

- Agent 생성 시 model, instructions, model_settings를 설정하는 방법
- @agent.instructions 데코레이터로 동적 instruction을 추가하는 방법
"""
import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

# --- Agent 정의 ---

agent = Agent(
    "openai:gpt-4o",
    instructions="도시 정보를 정확하게 알려줘.",
    model_settings={
        'temperature': 0.3,
        'max_tokens': 500,
    },
)

@agent.instructions
def length():
    return "결과는 500글자 이내로 작성해줘."

# --- 실행 ---

async def main():
    response = await agent.run(user_prompt="서울에 대해 알려줘")
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
