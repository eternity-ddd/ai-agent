"""
메시지 히스토리 없이 연속 호출하는 경우

- ChatPromptTemplate | llm | StrOutputParser 체인은 매 호출마다 독립적
- 두 번째 질문에서 "방금 전 도시"를 언급하지만, 이전 맥락이 없어 엉뚱한 답변을 함
- 올바른 멀티 턴 대화는 03-multi-turn.py 참고

대응하는 pydantic 예제: pydantic/01-basic/02-single-turn.py
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 체인 정의 ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0.3, max_tokens=500)

chain = prompt | llm | StrOutputParser()

# --- 실행 (히스토리 미전달 → 두 번째 질문에서 맥락 단절) ---

def main():
    response = chain.invoke({"question": "서울에 대해 알려줘"})
    print(response)

    response = chain.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"})
    print(response)

if __name__ == "__main__":
    main()
