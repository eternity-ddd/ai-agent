"""
create_agent에서 구조화된 출력 (response_format)

- create_agent의 response_format으로 에이전트의 최종 출력을 Pydantic BaseModel로 구조화
- 01-state-and-output.py의 with_structured_output()은 체인(파이프) 전용이고,
  response_format은 에이전트 전용 — 사용 방식이 다르므로 상황에 맞게 선택
- 도구를 사용하면서 동시에 구조화된 출력도 가능

대응하는 pydantic 예제: pydantic/02-deps-and-output/01-state-and-output.py
  (PydanticAI는 Agent의 output_type 하나로 체인/에이전트 구분 없이 동일하게 동작)
"""
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import create_agent

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

# --- Agent 정의 (구조화된 출력) ---

agent = create_agent(
    model="gpt-5.2",
    system_prompt=f"도시 정보를 정확히 알려줘. 결과는 {state.length} 글자 이내로 작성해줘.",
    response_format=CityInfo,
)

# --- 실행 ---

def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "서울에 대해 알려줘"}]}
    )
    # 구조화된 출력은 마지막 메시지의 content에 JSON으로 반환됨
    print(result["structured_response"])

if __name__ == "__main__":
    main()
