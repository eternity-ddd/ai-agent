"""
message_history를 활용한 멀티 턴 대화

- response.all_messages()로 이전 대화 내역을 가져와 다음 호출에 전달
- 에이전트가 이전 맥락("서울")을 기억하여 "방금 전 도시의 위도와 경도"에 정확히 답변
- 02-single-turn.py와 비교하여 히스토리 전달의 중요성을 확인
"""
import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

# --- Agent 정의 ---

agent = Agent(
    "openai:gpt-5.2",
    instructions="도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘.",
    model_settings={
        'temperature': 0.3,
        'max_tokens': 500,
    },
)

# --- 실행 (히스토리 전달 → 이전 맥락 유지) ---

async def main():
    response = await agent.run(user_prompt="서울에 대해 알려줘")
    print(response.output)

    response = await agent.run(
        user_prompt="방금 전 도시의 위도와 경도를 알려줘",
        message_history=response.all_messages(),
    )
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
