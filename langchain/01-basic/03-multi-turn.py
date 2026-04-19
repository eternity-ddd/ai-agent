"""
RunnableWithMessageHistory를 활용한 멀티 턴 대화

- ChatMessageHistory: 대화 히스토리를 자동으로 저장/관리하는 LangChain 내장 기능
- RunnableWithMessageHistory: 체인에 히스토리 관리 기능을 래핑
- session_id로 여러 대화 세션을 구분할 수 있음
- 수동으로 HumanMessage/AIMessage를 append할 필요 없이 자동으로 누적됨
- 02-single-turn.py와 비교하여 히스토리 전달의 중요성을 확인

대응하는 pydantic 예제: pydantic/01-basic/03-multi-turn.py
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# --- 체인 정의 (history 슬롯 포함) ---

# MessagePlaceHolder("history") 위치에 이전까지의 메시지들을 저장하도록 지정
prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

llm = init_chat_model(
    "gpt-5.2",
    model_provider="openai",
    temperature=0.3,
    max_tokens=500,
)

chain = prompt | llm | StrOutputParser()

# --- 히스토리 관리 (LangChain이 자동으로 처리) ---

store = {}

# SQLChatMessageHistory, FileChatMessageHistory, MongoDBChatMessageHistory 등을 제공
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --- 실행 (히스토리가 자동으로 누적됨) ---

def main():
    # 메시지를 저장하고 조회하는데 사용할 session_id 키를 "user-1"로 지정
    config = {"configurable": {"session_id": "user-1"}}

    response = chain_with_history.invoke({"question": "서울에 대해 알려줘"}, config=config)
    print(response)

    response = chain_with_history.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"}, config=config)
    print(response)

if __name__ == "__main__":
    main()
