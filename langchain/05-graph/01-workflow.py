"""
[LangGraph 도입 전] create_agent로 복잡한 워크플로우를 구현한 예제

워크플로우:
  사용자 입력 → 영화 추천 → 리뷰 검색 → 평점 확인 → (5점 이하면 종료, 초과면 반복)

이 예제가 보여주는 문제점:
- create_agent를 써도 분기/반복 로직은 main()의 if/while에 흩어져 있음
- 워크플로우가 복잡해질수록 main()이 비대해지고, 단계 추가/변경 시 전체 수정 필요
- 상태 영속화(일시 정지/재개)가 불가능 — 중간에 멈추면 처음부터 다시 시작
- 워크플로우 구조를 시각화할 방법이 없음

→ 이런 한계를 LangGraph가 해결함 (02-graph.py 참고)

대응하는 pydantic 예제: pydantic/05-graph/01-workflow.py
"""
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain.agents import create_agent
from rich.prompt import Prompt

load_dotenv()

# --- 출력 모델 ---

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

class ReviewScore(BaseModel):
    value: int = Field(description="리뷰 점수", ge=1, le=10)

# --- Agent 정의 ---

movie_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘",
    response_format=MovieOutput,
)

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 리뷰를 1개만 찾아줘.",
)

score_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="리뷰 점수는 1점에서 10점 사이 정수값으로 변환해서 줘",
    response_format=ReviewScore,
)

# --- 각 단계를 함수로 분리 ---

def find_movie(year: str) -> MovieOutput:
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": f"{year}년에 개봉한 영화를 추천해 줘"}]}
    )
    return result["structured_response"]

def review_movie(movie_title: str) -> str:
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰를 1개만 찾아줘"}]}
    )
    return result["messages"][-1].content

def review_score(movie_title: str) -> ReviewScore:
    result = score_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰 점수를 찾아줘"}]}
    )
    return result["structured_response"]

# --- 워크플로우 실행: create_agent를 써도 분기/반복은 수동 ---

def main():
    while True:
        year = Prompt.ask("영화 개봉 연도를 입력하세요")

        movie = find_movie(year)
        if movie.title is None:
            print("영화를 찾지 못해 종료합니다.")
            break
        print(f"영화 제목: {movie.title}")

        review = review_movie(movie.title)
        print(review)

        score = review_score(movie.title)
        if score.value <= 5:
            print("평점이 5점 이하인 경우 종료합니다.")
            break
        print(f"리뷰 점수: {score.value}")

if __name__ == "__main__":
    main()
