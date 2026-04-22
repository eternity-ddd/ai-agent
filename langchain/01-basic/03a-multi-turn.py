"""
SQLChatMessageHistory를 활용한 영속화된 멀티 턴 대화

- 03-multi-turn.py와 구조는 동일하지만 저장소만 SQLite로 교체
- get_session_history가 SQLChatMessageHistory를 반환하도록만 바꾸면 끝
- 프로세스를 재시작해도 같은 session_id면 이전 대화가 복원됨
"""
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "multi_turn.db")

# --- 체인 정의 (history 슬롯 포함) ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    temperature=0.3,
    max_tokens=500,
)

chain = prompt | llm | StrOutputParser()

# --- 히스토리 관리 (SQLite 영속화) ---

# 메모리와 달리 store dict가 필요 없음 — SQLite가 저장소 역할을 대신함
# 같은 session_id로 접근하면 DB에서 기존 메시지를 자동으로 로드
def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{DB_PATH}")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --- 실행 (프로세스를 재시작해도 히스토리가 유지됨) ---

def main():
    config = {"configurable": {"session_id": "user-1"}}

    response = chain_with_history.invoke({"question": "서울에 대해 알려줘"}, config=config)
    print(response)

    response = chain_with_history.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"}, config=config)
    print(response)

if __name__ == "__main__":
    main()
