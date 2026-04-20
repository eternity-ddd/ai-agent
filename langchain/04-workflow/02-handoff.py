"""
핸드오프(Handoff) 패턴: 에이전트 결과를 다음 에이전트에 순차 전달

- movie_agent → review_agent 순서로 호출하며, 에이전트 간 직접 연결 없음
- main()에서 에이전트 흐름을 결정하고, 결과를 추출하여 다음 에이전트에 전달
- create_agent를 써도 에이전트 간 연결은 수동 — 이것이 그래프가 필요한 이유

→ 체인(파이프)으로 구현하면 더 복잡해짐 (02a-handoff-chain.py 참고)

대응하는 pydantic 예제: pydantic/04-workflow/02-handoff.py
"""
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain.agents import create_agent

load_dotenv()

# --- 모델 정의 ---

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

# --- Agent 정의 ---

movie_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘",
    response_format=MovieOutput,
)

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 가장 최신에 작성된 리뷰를 선택해줘.",
)

# --- 실행: 에이전트를 써도 연결은 수동 ---

def main():
    # 1단계: 영화 추천
    movie_result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": "2020년에 나온 한국 영화를 추천해 줘"}]}
    )
    movie = movie_result["structured_response"]

    # 2단계: None 체크 — 에이전트를 써도 분기는 직접 처리
    if movie.title is None:
        print("영화를 찾지 못했습니다.")
        return

    print(f"영화 제목: {movie.title}")

    # 3단계: 리뷰 검색 — 에이전트 간 결과를 수동으로 연결
    review_result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie.title}인 영화의 리뷰를 찾아줘"}]}
    )
    print(review_result["messages"][-1].content)

if __name__ == "__main__":
    main()
