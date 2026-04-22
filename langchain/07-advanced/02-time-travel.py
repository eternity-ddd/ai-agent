"""
[시간여행] 체크포인트 이력 조회·편집·재실행

- graph.get_state_history(config): 스레드의 모든 체크포인트를 최신순으로 반환
- graph.update_state(checkpoint_config, values): 과거 시점 상태를 편집 (새 체크포인트 생성)
- graph.invoke(None, config=new_config): 편집된 시점부터 이어서 재실행

LangGraph 계열에만 존재하는 기능으로, 디버깅·실험(다른 분기 탐색)에 유용하다.

실행: python 02-time-travel.py
"""
import os
import sqlite3
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "time_travel.db")
THREAD_ID = "tt-1"

# --- 상태와 노드 ---

class State(TypedDict):
    counter: int
    note: str

def increment(state: State) -> State:
    new_value = state["counter"] + 1
    return {"counter": new_value, "note": f"incremented to {new_value}"}

def double(state: State) -> State:
    new_value = state["counter"] * 2
    return {"counter": new_value, "note": f"doubled to {new_value}"}

# --- 그래프 조립 ---
# START → inc → dbl → END

builder = StateGraph(State)
builder.add_node("inc", increment)
builder.add_node("dbl", double)
builder.add_edge(START, "inc")
builder.add_edge("inc", "dbl")
builder.add_edge("dbl", END)

# 매 실행마다 깨끗한 상태로 시작하기 위해 DB 초기화
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

graph = builder.compile(
    checkpointer=SqliteSaver(sqlite3.connect(DB_PATH, check_same_thread=False))
)

# --- 실행 ---

def main():
    config = {"configurable": {"thread_id": THREAD_ID}}

    # 1) 초기 실행: counter=5 → inc(6) → dbl(12)
    result = graph.invoke({"counter": 5, "note": "start"}, config=config)
    print(f"[1회차] 최종: {result}")

    # 2) 체크포인트 이력 조회 (최신순)
    print("\n[체크포인트 이력]")
    history = list(graph.get_state_history(config))
    for i, snapshot in enumerate(history):
        next_nodes = snapshot.next or "(END)"
        print(f"  {i}: values={snapshot.values}, next={next_nodes}")

    # 3) "inc 완료 / dbl 직전" 시점 체크포인트 찾기
    target = next(s for s in history if s.next == ("dbl",))
    print(f"\n[편집 대상 시점] values={target.values}, next={target.next}")

    # 4) 해당 시점 상태를 counter=100으로 덮어쓰고 새 체크포인트 생성
    new_config = graph.update_state(target.config, {"counter": 100})
    print(f"[편집 완료] 새 checkpoint_id={new_config['configurable']['checkpoint_id']}")

    # 5) 편집된 시점부터 재실행 (None = 기존 상태로 계속)
    # dbl이 실행되어 counter 100 → 200
    replayed = graph.invoke(None, config=new_config)
    print(f"\n[재실행] 최종: {replayed}")

if __name__ == "__main__":
    main()
