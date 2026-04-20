"""
구조화된 출력을 JSON으로 변환하는 방법

- JsonOutputParser: 프롬프트에 JSON 스키마를 주입(지시)하고, LLM 응답을 dict로 파싱
- with_structured_output()은 API 레벨에서 JSON을 강제하지만,
  JsonOutputParser는 프롬프트 텍스트로 유도하므로 LLM이 무시할 수 있음
- PydanticAI는 model_dump_json() 한 줄로 끝나지만,
  LangChain은 parser 생성 + format_instructions 주입 + 파이프 연결이 필요

대응하는 pydantic 예제: pydantic/02-deps-and-output/02-json-output.py
"""
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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

# --- 체인 정의 (프롬프트 | LLM | JSON 파서) ---

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 {length} 글자 이내로 작성해줘.\n{format_instructions}"),
    ("human", "{question}"),
])

llm = init_chat_model("gpt-4o", model_provider="openai")

parser = JsonOutputParser(pydantic_object=CityInfo)

chain = prompt | llm | parser

# --- 실행 ---

def main():
    # 전역 변수 state에 직접 접근
    # parser에 지정된 형식대로 응답을 생성하도록 format_instructions 지시
    response = chain.invoke({
        "length": state.length,
        "question": "서울에 대해 알려줘",
        "format_instructions": parser.get_format_instructions(),
    })
    print(response)

if __name__ == "__main__":
    main()
