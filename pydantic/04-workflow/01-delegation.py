"""
위임(Delegation) 패턴: 에이전트가 도구를 통해 다른 에이전트를 호출

- movie_agent가 영화를 선택한 뒤, @tool 내부에서 review_agent를 호출하여 리뷰를 검색
- 에이전트 간 의존성(deps)을 ctx.deps로 전달
- movie_agent가 워크플로우 전체를 주도하며, review_agent는 하위 도구로 동작
- 02-handoff.py의 핸드오프 패턴과 비교: 여기서는 에이전트가 직접 다른 에이전트를 호출
"""
import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import WebSearch

load_dotenv()

# --- 모델 정의 ---

@dataclass
class ReviewCriteria:
    criteria: str

# --- Agent 정의 ---

movie_agent = Agent[ReviewCriteria](
    'openai:gpt-5.2',
    deps_type=ReviewCriteria,   # type: ignore
    instructions="영화 전문가들을 위한 영화를 선택해줘",
)

review_agent = Agent[ReviewCriteria](
    "openai-responses:gpt-5.2",
    deps_type=ReviewCriteria,   # type: ignore
    capabilities=[WebSearch()],
)

# --- 도구 정의 (movie_agent가 review_agent에 위임) ---

@movie_agent.tool
async def review_movie(ctx: RunContext[ReviewCriteria], movie_title: str) -> str:
    """제목이 movie_title인 영화의 리뷰 찾기"""
    result = await review_agent.run(
        f"제목이 {movie_title}인 영화의 리뷰를 찾아줘. {ctx.deps.criteria} 조건에 해당하는 리뷰를 선택해줘.",
        deps=ctx.deps,
    )
    return result.output

# --- 실행 ---

async def main():
    result = await movie_agent.run(
        "2020년에 나온 한국 영화를 추천해 주고 그 영화의 리뷰를 알려줘",
        deps=ReviewCriteria(criteria="가장 최신에 작성된"),
    )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
