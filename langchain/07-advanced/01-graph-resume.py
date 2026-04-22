"""
[그래프 + 중단/재개] interrupt()와 checkpointer로 실행을 중단·재개

- 노드 안에서 interrupt()를 호출하면 그래프가 그 지점에서 일시 정지됨
- 체크포인터가 상태를 DB에 저장하므로 프로세스가 종료되어도 상태 복원 가능
- 같은 thread_id로 Command(resume=값)을 invoke하면 중단 지점부터 재개됨

이 스크립트는 두 단계를 **별도 프로세스**로 실행하도록 만들어졌다.
    python 01-graph-resume.py start     # 1차: interrupt에서 정지 후 종료
    python 01-graph-resume.py continue   # 2차: DB에서 상태를 읽어 재개

이 기능은 LangGraph 계열에만 존재한다 — 체인(RunnableWithMessageHistory)으로는 불가능.
처음에 어느 계열을 선택하느냐가 나중에 가능한 일을 결정한다.
"""
import os
import sys
import sqlite3
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command, interrupt

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "graph_resume.db")
THREAD_ID = "approval-1"

# --- 출력 모델 ---

class MovieOutput(BaseModel):
    title: str = Field(description="영화 제목")

# --- 상태 ---

class WorkflowState(TypedDict):
    year: str
    movie_title: str
    approved: bool
    review: str

# --- 에이전트 ---

movie_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 전문가야. 영화 하나를 추천해줘.",
    response_format=MovieOutput,
)

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 리뷰를 1개만 찾아줘.",
)

# --- 노드 ---

def recommend_node(state: WorkflowState) -> WorkflowState:
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": f"{state['year']}년 한국 영화를 추천해줘"}]}
    )
    title = result["structured_response"].title
    print(f"[recommend] 추천 영화: {title}")
    return {"movie_title": title}

def approval_node(state: WorkflowState) -> WorkflowState:
    # interrupt가 호출되는 순간 그래프 상태가 checkpointer(SQLite)에 저장되고
    # graph.invoke()는 __interrupt__를 결과에 담아 반환한다.
    # 다음 프로세스에서 graph.invoke(Command(resume=값), config=...)로 재개하면
    # 그 값이 answer에 들어오고 노드가 이어서 실행된다.
    answer = interrupt(f"'{state['movie_title']}' 영화의 리뷰를 찾을까요? (yes/no)")
    return {"approved": answer == "yes"}

def review_node(state: WorkflowState) -> WorkflowState:
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"{state['movie_title']} 리뷰를 찾아줘"}]}
    )
    review = result["messages"][-1].content
    print(f"[review] {review[:200]}...")
    return {"review": review}

# --- 분기 ---

def should_fetch_review(state: WorkflowState) -> str:
    return "review" if state["approved"] else END

# --- 그래프 조립 ---
# 주의: 그래프 구조(노드 이름/엣지)는 start와 continue 두 프로세스에서 동일해야 한다.
# checkpointer가 저장하는 것은 "상태 + 실행 위치"일 뿐, 그래프 자체는 아니기 때문이다.

def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(WorkflowState)
    builder.add_node("recommend", recommend_node)
    builder.add_node("approval", approval_node)
    builder.add_node("review", review_node)

    builder.add_edge(START, "recommend")
    builder.add_edge("recommend", "approval")
    builder.add_conditional_edges("approval", should_fetch_review)
    builder.add_edge("review", END)

    return builder.compile(checkpointer=checkpointer)

# --- 1단계: 시작 후 interrupt에서 정지 ---

def start():
    """새 thread_id로 그래프를 시작하고 approval 노드의 interrupt에서 정지한다."""
    graph = build_graph(SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False)))
    config = {"configurable": {"thread_id": THREAD_ID}}

    result = graph.invoke(
        {"year": "2020", "movie_title": "", "approved": False, "review": ""},
        config=config,
    )
    print("중단 메시지:", result["__interrupt__"])
    print(f"\n상태가 {DB_PATH}에 저장되었습니다.")
    print(f"`python {os.path.basename(__file__)} continue`로 재개하세요.")

# --- 2단계: 별도 프로세스에서 DB 상태를 읽어 재개 ---

def resume():
    """DB에 저장된 중단 지점부터 실행을 재개한다. 같은 thread_id를 사용해야 한다."""
    graph = build_graph(SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False)))
    config = {"configurable": {"thread_id": THREAD_ID}}

    # DB에 저장된 상태를 읽어와 확인
    snapshot = graph.get_state(config)
    print(f"복원된 상태: {snapshot.values}")
    print(f"다음 실행 예정 노드: {snapshot.next}")

    # 중단 지점에서 interrupt()가 던졌던 원래 질문을 복원해 출력
    pending_interrupts = [intr for task in snapshot.tasks for intr in task.interrupts]
    if pending_interrupts:
        print(f"\n질문: {pending_interrupts[0].value}")

    answer = input("답변 (yes/no): ").strip()

    # 초기 입력 대신 Command(resume=...)를 넘기면 approval 노드부터 재개
    final = graph.invoke(Command(resume=answer), config=config)
    print("\n최종 상태:", final)

# --- 진입점 ---

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "start"
    if mode == "start":
        start()
    elif mode == "continue":
        resume()
    else:
        print(f"Usage: python {os.path.basename(__file__)} [start|continue]")
        sys.exit(1)

if __name__ == "__main__":
    main()
