"""
[LangGraph 버전] 01-workflow.py와 동일한 워크플로우를 LangGraph로 구현

워크플로우:
  사용자 입력 → 영화 추천 → 리뷰 검색 → 평점 확인 → (5점 이하면 종료, 초과면 반복)

01-workflow.py 대비 개선점:
- 각 단계가 독립된 노드 함수로 분리되어 흐름이 명확함
- 분기/반복이 conditional_edges로 선언되어 구조가 코드에 드러남
- graph.get_graph().draw_mermaid()로 워크플로우를 시각화할 수 있음
- TypedDict state로 노드 간 데이터 공유가 구조화됨

대응하는 pydantic 예제: pydantic/05-graph/02-graph.py
"""
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from rich.prompt import Prompt

load_dotenv()

# --- State: 노드 간 공유되는 변경 가능한 데이터 ---

class WorkflowState(TypedDict):
    year: str
    movie_title: str
    review: str
    score: int

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

# --- 노드 정의 ---

def input_node(state: WorkflowState) -> WorkflowState:
    """사용자로부터 영화 개봉 연도를 입력받는 노드"""
    year = Prompt.ask("영화 개봉 연도를 입력하세요")
    return {"year": year}

def find_movie_node(state: WorkflowState) -> WorkflowState:
    """movie_agent를 이용해 영화를 추천받는 노드"""
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": f"{state['year']}년에 개봉한 영화를 추천해 줘"}]}
    )
    movie = result["structured_response"]
    title = movie.title if movie.title else ""
    print(f"영화 제목: {title}" if title else "영화를 찾지 못했습니다.")
    return {"movie_title": title}

def review_movie_node(state: WorkflowState) -> WorkflowState:
    """review_agent를 이용해 영화 리뷰를 검색하는 노드"""
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {state['movie_title']}인 영화의 리뷰를 1개만 찾아줘"}]}
    )
    review = result["messages"][-1].content
    print(review)
    return {"review": review}

def check_score_node(state: WorkflowState) -> WorkflowState:
    """score_agent를 이용해 평점을 확인하는 노드"""
    result = score_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {state['movie_title']}인 영화의 리뷰 점수를 찾아줘"}]}
    )
    score = result["structured_response"]
    print(f"리뷰 점수: {score.value}")
    return {"score": score.value}

# --- 분기 조건 ---

def check_movie_found(state: WorkflowState) -> str:
    """영화를 찾았는지 확인"""
    return "review_movie" if state["movie_title"] else END

def check_score_value(state: WorkflowState) -> str:
    """평점이 5점 초과인지 확인"""
    return "input" if state["score"] > 5 else END

# --- 그래프 조립 ---

graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("input", input_node)
graph_builder.add_node("find_movie", find_movie_node)
graph_builder.add_node("review_movie", review_movie_node)
graph_builder.add_node("check_score", check_score_node)

graph_builder.add_edge(START, "input")
graph_builder.add_edge("input", "find_movie")
graph_builder.add_conditional_edges("find_movie", check_movie_found)
graph_builder.add_edge("review_movie", "check_score")
graph_builder.add_conditional_edges("check_score", check_score_value)

movie_graph = graph_builder.compile()

# --- 실행 ---

def main():
    print(movie_graph.get_graph().draw_mermaid())
    print("---")

    result = movie_graph.invoke({
        "year": "",
        "movie_title": "",
        "review": "",
        "score": 0,
    })
    print(result)

if __name__ == "__main__":
    main()
