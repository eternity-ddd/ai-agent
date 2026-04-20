"""
[안티패턴] message_history 없이 연속 호출하는 경우

- 두 번째 질문에서 "방금 전 도시"를 언급하지만, 히스토리를 전달하지 않으므로
  에이전트는 이전 대화 맥락을 알 수 없어 엉뚱한 답변을 하거나 오류가 발생함
- 올바른 멀티 턴 대화는 03-multi-turn.py 참고
"""
import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent

load_dotenv()

# --- Agent 정의 ---

agent = Agent(
    "openai:gpt-4o",
    instructions="도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘.",
    model_settings={
        'temperature': 0.3,
        'max_tokens': 500,
    },
)

# --- 실행 (히스토리 미전달 → 두 번째 질문에서 맥락 단절) ---

async def main():
    response = await agent.run(user_prompt="서울에 대해 알려줘")
    print(response.output)

    response = await agent.run(user_prompt="방금 전 도시의 위도와 경도를 알려줘")
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
