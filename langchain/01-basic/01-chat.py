"""
ChatModel 기본 사용법과 LCEL 파이프(|) 체이닝

- ChatPromptTemplate | llm | StrOutputParser 파이프로 체인을 구성하는 방법
- 파이프(|)는 LCEL(LangChain Expression Language)의 핵심으로,
  프롬프트 → LLM → 출력 파서를 선언적으로 연결

대응하는 pydantic 예제: pydantic/01-basic/01-chat.py
"""
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 체인 정의 (프롬프트 | LLM | 출력파서) ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm = init_chat_model(
    "gpt-4o",
    model_provider="openai",
    temperature=0.3,
    max_tokens=500,
)

# 반환된 AIMessage에서 텍스트 결과만 파싱
parser = StrOutputParser()

# response = prompt.invoke({"question": "서울에 대해 알려줘"})
# response = llm.invoke(response)
# response = parser.invoke(response)

chain = prompt | llm | parser

# --- 실행 ---

def main():
    response = chain.invoke({"question": "서울에 대해 알려줘"})
    print(response)

if __name__ == "__main__":
    main()
