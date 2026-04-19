"""
핸드오프(Handoff) 패턴: 에이전트 결과를 다음 에이전트에 순차 전달

- find_movie() → review_movie() 순서로 호출하며, 에이전트 간 직접 연결 없음
- movie_agent의 구조화된 출력(MovieOutput)을 받아 review_agent에 문자열로 전달
- 01-delegation.py와 비교: 여기서는 오케스트레이션 로직이 main()에 있고,
  에이전트끼리는 서로의 존재를 모름
"""
import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import Field, BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

load_dotenv()

# --- 모델 정의 ---

@dataclass
class ReviewCriteria:
    criteria: str

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

# --- Agent 정의 ---

movie_agent = Agent[None, MovieOutput](
    'openai:gpt-5.2',
    output_type=MovieOutput,    # type: ignore
    instructions="영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘",
)

review_agent = Agent[ReviewCriteria](
    "openai-responses:gpt-5.2",
    deps_type=ReviewCriteria,   # type: ignore
    capabilities=[WebSearch()],
)

# --- 각 단계를 함수로 분리 ---

async def find_movie() -> MovieOutput:
    result = await movie_agent.run("2020년에 나온 한국 영화를 추천해 줘")
    return result.output

async def review_movie(movie_title: str) -> str:
    result = await review_agent.run(
        f"제목이 {movie_title}인 영화의 리뷰를 찾아줘",
        deps=ReviewCriteria(criteria="가장 최신에 작성된"),
    )
    return result.output

# --- 실행 ---

async def main():
    movie = await find_movie()
    if movie.title is not None:
        review = await review_movie(movie.title)
        print(review)

if __name__ == "__main__":
    asyncio.run(main())
