"""
[Graph 버전] 01-workflow.py와 동일한 워크플로우를 pydantic-graph로 구현

워크플로우:
  사용자 입력 → 영화 추천 → 리뷰 검색 → 평점 확인 → (5점 이하면 종료, 초과면 반복)

01-workflow.py 대비 개선점:
- 각 단계가 독립된 노드로 분리되어 흐름이 명확함
- 분기/반복이 리턴 타입으로 선언되어 구조가 코드에 드러남
- graph.mermaid_code()로 워크플로우를 시각화할 수 있음
- state로 노드 간 데이터 공유가 구조화됨
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import Field, BaseModel
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from rich.prompt import Prompt

load_dotenv()

# --- State: 노드 간 공유되는 변경 가능한 데이터 ---

@dataclass
class WorkflowState:
    year: str = ""
    movie_title: str = ""
    review: str = ""
    score: int = 0

# --- 출력 모델 ---

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

class ReviewScore(BaseModel):
    value: int = Field(description="리뷰 점수", ge=1, le=10)

# --- Agent 정의 ---

movie_agent = Agent[None, MovieOutput](
    'openai:gpt-5.2',
    output_type=MovieOutput,    # type: ignore
    instructions="영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘",
)

review_agent = Agent(
    "openai-responses:gpt-5.2",
    capabilities=[WebSearch()],
)

score_agent = Agent[None, ReviewScore](
    'openai:gpt-5.2',
    output_type=ReviewScore,
    instructions="리뷰 점수는 1점에서 10점 사이 정수값으로 변환해서 줘",
)

# --- 노드 정의: 리턴 타입이 곧 다음 노드(엣지)를 결정 ---

@dataclass
class InputNode(BaseNode[WorkflowState, None, str]):
    """사용자로부터 영화 개봉 연도를 입력받는 노드"""
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> FindMovieNode:
        ctx.state.year = Prompt.ask("영화 개봉 연도를 입력하세요")
        return FindMovieNode()

@dataclass
class FindMovieNode(BaseNode[WorkflowState, None, str]):
    """movie_agent를 이용해 영화를 추천받는 노드"""
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> ReviewMovieNode | End[str]:
        result = await movie_agent.run(
            f"{ctx.state.year}년에 개봉한 영화를 추천해 줘")

        if result.output.title is None:
            return End("영화를 찾지 못해 종료합니다.")

        ctx.state.movie_title = result.output.title

        print(f"영화 제목: {ctx.state.movie_title}")
        return ReviewMovieNode()

@dataclass
class ReviewMovieNode(BaseNode[WorkflowState, None, str]):
    """review_agent를 이용해 영화 리뷰를 검색하는 노드"""
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> CheckScoreNode:
        result = await review_agent.run(
            f"제목이 {ctx.state.movie_title}인 영화의 리뷰를 1개만 찾아줘")

        ctx.state.review = result.output

        print(ctx.state.review)
        return CheckScoreNode()

@dataclass
class CheckScoreNode(BaseNode[WorkflowState, None, str]):
    """score_agent를 이용해 평점을 확인하고, 5점 이하면 종료 / 초과면 반복"""
    async def run(self, ctx: GraphRunContext[WorkflowState]) -> InputNode | End[str]:
        result = await score_agent.run(
            f"제목이 {ctx.state.movie_title}인 영화의 리뷰 점수를 찾아줘")

        ctx.state.score = result.output.value
        if ctx.state.score <= 5:
            return End("평점이 5점 이하인 경우 종료합니다.")
        print(f"리뷰 점수: {ctx.state.score}")

        return InputNode()

# --- 그래프 조립 및 실행 ---

movie_graph = Graph(
    nodes=(InputNode, FindMovieNode, ReviewMovieNode, CheckScoreNode)
)

async def main():
    print(movie_graph.mermaid_code())
    print("---")

    state = WorkflowState()
    result = await movie_graph.run(InputNode(), state=state)
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
