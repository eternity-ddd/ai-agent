"""
PydanticAI 내장 capabilities 사용법

- Thinking: 모델의 사고 과정(chain-of-thought)을 활성화
- WebSearch: 웹 검색 기능을 에이전트에 부여
- WebFetch: 특정 도메인의 웹 페이지를 직접 가져오는 기능 (allowed_domains로 허용 도메인 제한)
- 직접 도구를 정의하지 않아도 내장 기능만으로 웹 검색/크롤링이 가능한 예제
"""
import asyncio

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch, WebFetch

load_dotenv()

# --- Agent 정의 ---

agent = Agent(
    "anthropic:claude-opus-4-6",
    instructions="도시 정보를 정확히 알려줘",
    capabilities=[
        Thinking(),
        WebSearch(),
        WebFetch(allowed_domains=['aladin.co.kr']),
    ],
)

# --- 실행 ---

async def main():
    response = await agent.run(
        user_prompt="오늘 삼성전자 주가를 조회해주고, aladin.co.kr에서 프로젝트 헤일메리 책 페이지를 찾아서 책 소개 부분을 요약해줘"
    )
    print(response.output)

if __name__ == "__main__":
    asyncio.run(main())
