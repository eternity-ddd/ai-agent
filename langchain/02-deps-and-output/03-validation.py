"""
Pydantic field_validator를 활용한 입력/출력 검증

- 입력(MyState), 출력(CityInfo) 모두에 field_validator를 적용하는 것은 PydanticAI와 동일
- 단, PydanticAI는 retries 옵션으로 출력 검증 실패 시 LLM에 재시도를 요청할 수 있지만,
  LangChain의 with_structured_output()은 재시도 메커니즘이 없어 직접 구현해야 함
- 아래 예제는 length=1000000을 전달하므로 validator에 의해 즉시 ValidationError 발생 (의도된 실패 예제)

대응하는 pydantic 예제: pydantic/02-deps-and-output/03-validation.py
"""
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 모델 정의 (validator 포함) ---

class MyState(BaseModel):
    length: int

    @field_validator('length')
    @classmethod
    def check_length(cls, v: int) -> int:
        if v >= 10000:
            raise ValueError("길이는 10000보다 작아야 한다.")
        return v

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

    @field_validator('population')
    @classmethod
    def check_population(cls, v: int) -> int:
        if v >= 100000:
            raise ValueError("인구는 100000보다 작아야 한다.")
        return v

# --- 체인 정의 (프롬프트 | LLM + 구조화 출력) ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 {length} 글자 이내로 작성해줘."),
    ("human", "{question}"),
])

# 출력 형식은 CityInfo로 지정
llm = init_chat_model("gpt-4o", model_provider="openai").with_structured_output(CityInfo)

chain = prompt | llm

# --- 실행 (의도된 실패: length=1000000은 validator에 의해 거부됨) ---

def main():
    # 의도된 실패: MyState의 validator가 length >= 10000을 거부
    state = MyState(length=1000000)
    response = chain.invoke({"length": state.length, "question": "서울에 대해 알려줘"})
    print(response.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
