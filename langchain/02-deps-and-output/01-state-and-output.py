"""
with_structured_output을 사용한 구조화된 출력

- with_structured_output(CityInfo): LLM 응답을 Pydantic BaseModel로 구조화
- LangChain에서 구조화된 출력은 모델 레벨 기능이므로 create_agent보다 체인(파이프)이 적합
  (create_agent에는 with_structured_output()을 직접 연결할 수 없음)
- LangChain에는 PydanticAI의 deps(의존성 주입)가 없어 전역 변수로 상태를 관리해야 함
- 상태값을 프롬프트에 전달할 때 문자열 키({length})를 사용하므로 오타 시 런타임 에러 발생
  → PydanticAI는 RunContext[MyState]로 타입 안전하게 접근

대응하는 pydantic 예제: pydantic/02-deps-and-output/01-state-and-output.py
"""
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 모델 정의 ---

@dataclass
class MyState:
    length: int

# 전역 변수로 상태 지정
state = MyState(200)

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

# --- 체인 정의 (프롬프트 | LLM + 구조화 출력) ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 {length} 글자 이내로 작성해줘."),
    ("human", "{question}"),
])

# 출력 형식은 CityInfo로 지정
llm = init_chat_model("gpt-5.2", model_provider="openai").with_structured_output(CityInfo)

chain = prompt | llm

# --- 실행 ---

def main():
    # 전역 변수 state에 직접 접근
    response = chain.invoke({"length": state.length, "question": "서울에 대해 알려줘"})
    print(response)

if __name__ == "__main__":
    main()
