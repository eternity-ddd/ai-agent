# LangChain

(+) **풍부한 built-in 컴포넌트** — RAG, 히스토리 관리, 벡터 DB 등 인프라성 기능 내장       
(+) **히스토리 자동 관리** — `RunnableWithMessageHistory` + `session_id`로 영속화까지       
(+) **LCEL 파이프(`|`)** — 구성 요소를 선언적으로 조합       
(+) **두 가지 실행 방식** — 모델 직접(`init_chat_model`) vs 에이전트(`create_agent`) 상황별 선택       
(+) **LangGraph의 고급 실행 제어** — checkpointer로 상태 영속화, `interrupt()`/`Command(resume)`로 중단·재개·HITL, 시간여행 지원       
(-) **높은 추상화** — 상황별로 적합한 컴포넌트를 익혀야 하고 디버깅이 어려움       
(-) **타입 안전성 부족** — 상태 주입이 `{length}` 같은 문자열 키 기반, 오타 시 런타임 에러       
(-) **방식의 이원화** — 체인(`with_structured_output`)과 에이전트(`response_format`)가 별도 방식       
(-) **비일관성** — 같은 목적(구조화 출력, 도구, 히스토리 등)에 공존하는 API가 여러 개

## 목차

- [1. 기본](#1-기본)
  - [1-1. 채팅](#1-1-채팅)
  - [1-2. 싱글 턴(Single-turn)](#1-2-싱글-턴single-turn)
  - [1-3. 멀티 턴(Multi-turn)](#1-3-멀티-턴multi-turn)
- [2. 의존성과 출력 관리](#2-의존성과-출력-관리)
  - [2-1. 상태(의존성)와 출력](#2-1-상태의존성와-출력)
  - [2-2. JSON 출력](#2-2-json-출력)
  - [2-2a. 에이전트에서 구조화된 출력](#2-2a-에이전트에서-구조화된-출력)
  - [2-3. Pydantic Validation 체크](#2-3-pydantic-validation-체크)
- [3. Tool을 이용한 Agent 구현](#3-tool을-이용한-agent-구현)
  - [3-1. Tool 등록](#3-1-tool-등록)
  - [3-2. Capability / 내장 도구](#3-2-capability--내장-도구)
- [4. 워크플로우 (멀티에이전트)](#4-워크플로우-멀티에이전트)
  - [4-1. 위임 (Delegation)](#4-1-위임-delegation)
  - [4-2. 핸드오프 (Handoff)](#4-2-핸드오프-handoff)
  - [4-3. 체이닝을 이용한 핸드오프 구현](#4-3-체이닝을-이용한-핸드오프-구현)
- [5. 그래프](#5-그래프)
  - [5-1. 복잡한 워크플로우의 문제점](#5-1-복잡한-워크플로우의-문제점)
  - [5-2. LangGraph로 해결](#5-2-langgraph로-해결)
- [6. RAG](#6-rag)
- [7.평가 : 비일관성과 강력함의 혼재](#7평가--비일관성과-강력함의-혼재)
  - [7-1. 구조화된 출력](#7-1-구조화된-출력)
  - [7-2. 도구 등록](#7-2-도구-등록)
  - [7-3. 히스토리 관리](#7-3-히스토리-관리)
  - [7-4. 에이전트 생성](#7-4-에이전트-생성)
  - [7-5. 패키지 분할](#7-5-패키지-분할)
  - [7-6. LangGraph의 고급 실행 제어](#7-6-langgraph의-고급-실행-제어)
  - [7-7. 현재의 상태(AI 의견)](#7-7-현재의-상태ai-의견)

---

## 1. 기본

LangChain에서는 `init_chat_model`로 LLM을 생성하고, `ChatPromptTemplate`으로 프롬프트를 정의하여 파이프(`|`)로 연결한다.

### 1-1. 채팅

[01-basic/01-chat.py](../../langchain/01-basic/01-chat.py)

**체인 구성**

LangChain의 핵심은 LCEL(LangChain Expression Language)의 파이프(`|`)이다.
프롬프트, LLM, 출력 파서를 선언적으로 연결한다.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm = init_chat_model(
    "gpt-4o", 
    model_provider="openai", 
    temperature=0.3, 
    max_tokens=500)

parser = StrOutputParser()
chain = prompt | llm | parser
```
**실행**

```python
response = chain.invoke({"question": "서울에 대해 알려줘"})
```

파이프 없이 각 Runnable의 `invoke()`를 수동으로 호출해도 동일한 결과를 얻을 수 있다.

```python
response = prompt.invoke({"question": "서울에 대해 알려줘"})
response = llm.invoke(response)
response = parser.invoke(response)
...
```



**PydanticAI와 비교**

| | PydanticAI | LangChain |
|---|---|---|
| LLM 호출 | `agent.run(user_prompt="...")` | `chain.invoke({"question": "..."})` |
| 시스템 메시지 | `instructions="..."` | `("system", "...")` in ChatPromptTemplate |
| 동적 instruction | `@agent.instructions` 데코레이터 | 프롬프트 변수 `{question}` |
| 출력 처리 | `response.output` (자동) | `StrOutputParser()` 필요 |

**전체 코드**

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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

parser = StrOutputParser()

chain = prompt | llm | parser

def main():
    response = chain.invoke({"question": "서울에 대해 알려줘"})
    print(response)

if __name__ == "__main__":
    main()
```

### 1-2. 싱글 턴(Single-turn)

[01-basic/02-single-turn.py](../../langchain/01-basic/02-single-turn.py)

PydanticAI와 동일하게, 히스토리를 전달하지 않으면 각 호출이 독립적이다.

```python
response = chain.invoke({"question": "서울에 대해 알려줘"})

# 히스토리 미전달 → "방금 전 도시"를 모름
response = chain.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"})
```

```
결과:
서울특별시는 대한민국의 수도이자 최대 도시로...

어떤 도시를 말씀하시는지 정보가 없습니다(이 대화에 도시명이 나오지 않았어요).
```

**전체 코드**

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘. 결과는 500글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm = init_chat_model(
    "gpt-4o", 
    model_provider="openai", 
    temperature=0.3, 
    max_tokens=500)

chain = prompt | llm | StrOutputParser()

def main():
    response = chain.invoke({"question": "서울에 대해 알려줘"})
    print(response)

    response = chain.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"})
    print(response)

if __name__ == "__main__":
    main()
```

### 1-3. 멀티 턴(Multi-turn)

[01-basic/03-multi-turn.py](../../langchain/01-basic/03-multi-turn.py)

PydanticAI는 `message_history=response.all_messages()`로 히스토리를 수동 전달하지만,
LangChain은 `RunnableWithMessageHistory`로 **자동 누적**할 수 있다.

**MessagesPlaceholder로 히스토리 슬롯 정의**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확하게 알려줘."),
    MessagesPlaceholder("history"),   # ← RunnableWithMessageHistory가 session_id로 history를 조회해 이 자리에 주입
    ("human", "{question}"),
])
```

**RunnableWithMessageHistory로 자동 관리**

`session_id`로 여러 사용자의 대화를 분리 관리할 수 있다.

**세션 저장소 정의**

`session_id`를 키로 히스토리 객체를 보관하는 `store`와 `session_id`별로 메시지를 반환하는 `get_session_history` 함수를 추가한다.

```python
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

**체인을 히스토리 기능으로 래핑**

기존 체인에 `RunnableWithMessageHistory`를 씌워 히스토리를 자동 읽기/쓰기 하게 만든다. `history_messages_key`는 프롬프트의 `MessagesPlaceholder("history")`로 주입될 이전 메시지 자리, `input_messages_key`는 히스토리에 append될 사용자 입력 키다.

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)
```

**호출 — session_id로 대화 세션 구분**

`config`에 `session_id`를 넣어 호출하면 같은 세션의 히스토리가 자동 누적되어 두 번째 질문에서 "방금 전 도시"를 기억한다.

```python
config = {"configurable": {"session_id": "user-1"}}
response = chain_with_history.invoke({"question": "서울에 대해 알려줘"}, config=config)
response = chain_with_history.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"}, config=config)
```

**PydanticAI와 비교**

| | PydanticAI | LangChain |
|---|---|---|
| 히스토리 전달 | `message_history=response.all_messages()` (수동) | `RunnableWithMessageHistory` (자동) |
| 세션 관리 | 없음 (직접 구현) | `session_id`로 내장 지원 |
| 영속화 | 없음 | `SQLChatMessageHistory`, `RedisChatMessageHistory` 등 교체 가능 |

**전체 코드**

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

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

store = {}

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

def main():
    config = {"configurable": {"session_id": "user-1"}}

    response = chain_with_history.invoke({"question": "서울에 대해 알려줘"}, config=config)
    print(response)

    response = chain_with_history.invoke({"question": "방금 전 도시의 위도와 경도를 알려줘"}, config=config)
    print(response)

if __name__ == "__main__":
    main()
```

**SQLite로 영속화** ([01-basic/03a-multi-turn.py](../../langchain/01-basic/03a-multi-turn.py))

`InMemoryChatMessageHistory`를 `SQLChatMessageHistory`로 교체만 하면 프로세스를 재시작해도 같은 `session_id`로 대화가 이어진다.
래퍼(`RunnableWithMessageHistory`) 구조는 그대로이고, `get_session_history`가 반환하는 타입만 바꾸면 된다.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

DB_PATH = "multi_turn.db"

def get_session_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(session_id=session_id, connection=f"sqlite:///{DB_PATH}")

# RunnableWithMessageHistory 래핑과 invoke 호출은 03-multi-turn.py와 동일
```

SQLite가 세션별 메시지를 영속화하기 때문에 메모리 버전의 `store`는 더 이상 필요하지 않다.

---

## 2. 의존성과 출력 관리

### 2-1. 상태(의존성)와 출력

[02-deps-and-output/01-deps-and-output.py](../../langchain/02-deps-and-output/01-deps-and-output.py)

**구조화된 출력**

`with_structured_output(CityInfo)`로 LLM 응답을 Pydantic BaseModel로 구조화한다.

```python
llm = init_chat_model("gpt-4o", model_provider="openai").with_structured_output(CityInfo)
chain = prompt | llm
```

**상태(의존성) 관리**

LangChain에는 PydanticAI의 `deps`가 없어 전역 변수로 상태를 관리하고,
프롬프트 변수(`{length}`)로 전달해야 한다.

```python
@dataclass
class MyState:
    length: int

state = MyState(200)  # 전역 변수

response = chain.invoke({"length": state.length, "question": "서울에 대해 알려줘"})
```

**PydanticAI와 비교**

| | PydanticAI | LangChain |
|---|---|---|
| 구조화된 출력 | `output_type=CityInfo` | `.with_structured_output(CityInfo)` |
| 상태 주입 | `deps=MyState(200)` (타입 안전) | 전역 변수 + 문자열 키 `{length}` (오타 시 런타임 에러) |
| 상태 접근 | `ctx.deps.length` | `state.length` (전역 변수 직접 참조) |

**전체 코드**

```python
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

@dataclass
class MyState:
    length: int

state = MyState(200)

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 {length} 글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm = init_chat_model("gpt-4o", model_provider="openai").with_structured_output(CityInfo)

chain = prompt | llm

def main():
    response = chain.invoke({"length": state.length, "question": "서울에 대해 알려줘"})
    print(response)

if __name__ == "__main__":
    main()
```

### 2-2. JSON 출력

[02-deps-and-output/02-json-output.py](../../langchain/02-deps-and-output/02-json-output.py)

`JsonOutputParser`는 두 가지 역할을 한다:
- LLM에게 "이 JSON 형식으로 답해"라고 **지시** (프롬프트에 삽입)
- LLM 응답을 실제로 dict로 **파싱**

```python
parser = JsonOutputParser(pydantic_object=CityInfo)
chain = prompt | llm | parser
```

**반환 타입의 차이**

| | 반환 타입 |
|---|---|
| `with_structured_output(CityInfo)` | Pydantic 인스턴스 (`CityInfo`) |
| `JsonOutputParser(pydantic_object=CityInfo)` | dict (JSON 파싱 결과) |

**PydanticAI와 비교**

PydanticAI는 `model_dump_json()` 한 줄로 끝나지만,
LangChain은 parser 생성 + `format_instructions` 주입 + 파이프 연결이 필요하다.

**전체 코드**

```python
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

@dataclass
class MyState:
    length: int

state = MyState(200)

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 {length} 글자 이내로 작성해줘.\n{format_instructions}"),
    ("human", "{question}"),
])

llm = init_chat_model("gpt-4o", model_provider="openai")

parser = JsonOutputParser(pydantic_object=CityInfo)

chain = prompt | llm | parser

def main():
    response = chain.invoke({
        "length": state.length,
        "question": "서울에 대해 알려줘",
        "format_instructions": parser.get_format_instructions(),
    })
    print(response)

if __name__ == "__main__":
    main()
```

### 2-2a. 에이전트에서 구조화된 출력

[02-deps-and-output/02a-agent-output.py](../../langchain/02-deps-and-output/02a-agent-output.py)

뒤에서 살펴볼 `create_agent`에서는 `response_format`으로 구조화된 출력을 지정한다.

```python
agent = create_agent(
    model="gpt-4o",
    system_prompt="...",
    response_format=CityInfo,
)

result = agent.invoke({"messages": [...]})
print(result["structured_response"])  # CityInfo 객체
```

**PydanticAI와 비교**

PydanticAI는 `output_type` 하나로 체인/에이전트 구분 없이 동일하게 동작하지만,
LangChain은 체인(`with_structured_output`)과 에이전트(`response_format`)로 방법이 나뉜다.

**전체 코드**

```python
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.agents import create_agent

load_dotenv()

@dataclass
class MyState:
    length: int

state = MyState(200)

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

agent = create_agent(
    model="gpt-4o",
    system_prompt=f"도시 정보를 정확히 알려줘. 결과는 {state.length} 글자 이내로 작성해줘.",
    response_format=CityInfo,
)

def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "서울에 대해 알려줘"}]}
    )
    print(result["structured_response"])

if __name__ == "__main__":
    main()
```

### 2-3. Pydantic Validation 체크

[02-deps-and-output/03-validation.py](../../langchain/02-deps-and-output/03-validation.py)

입력/출력에 `field_validator`를 적용하는 것은 PydanticAI와 동일하다.

**PydanticAI와 비교**

| | PydanticAI | LangChain |
|---|---|---|
| 검증 방식 | 동일 (Pydantic field_validator) | 동일 |
| 출력 검증 실패 시 재시도 | `retries=3`으로 LLM에 자동 재시도 | 재시도 메커니즘 없음, 직접 구현 필요 |

**전체 코드**

```python
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘. 결과는 {length} 글자 이내로 작성해줘."),
    ("human", "{question}"),
])

llm = init_chat_model("gpt-4o", model_provider="openai").with_structured_output(CityInfo)

chain = prompt | llm

def main():
    state = MyState(length=1000000)
    response = chain.invoke({"length": state.length, "question": "서울에 대해 알려줘"})
    print(response.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
```

---

## 3. Tool을 이용한 Agent 구현

에이전트에 도구를 등록하면 LLM이 필요한 도구를 스스로 판단해 호출한다.
"생각 → 도구 호출 → 관찰 → 다시 생각" 루프가 바로 **ReAct 패턴**이며, LangChain에서는 `create_agent`가 이 루프를 내부에서 자동 수행한다.

### 3-1. Tool 등록

[03-tool/01-tool.py](../../langchain/03-tool/01-tool.py) — 수동 실행  
[03-tool/01a-executor.py](../../langchain/03-tool/01a-executor.py) — AgentExecutor (레거시)  
[03-tool/01b-agent.py](../../langchain/03-tool/01b-agent.py) — create_agent (최신 권장)

LangChain에서 도구를 등록하는 방식은 세 가지가 공존한다.

**bind_tools() — 수동 실행 (01-tool.py)**

`bind_tools()`는 LLM에게 도구 스키마만 전달할 뿐, 실제 실행은 하지 않는다.
개발자가 `tool_calls`를 꺼내서 직접 실행하는 루프를 구현해야 한다.

```python
@tool(args_schema=Search)
def web_search(keyword: str, location: str) -> list[str]:
    """웹 검색을 해서 데이터를 가져옵니다."""
    return ["남산타워", "청와대", "글라스 하우스"]

chain = prompt | init_chat_model("gpt-4o", model_provider="openai").bind_tools(tools)

# 도구 호출 루프를 직접 구현
response = chain.invoke(...)
for tool_call in response.tool_calls:
    result = tools_map[tool_call["name"]].invoke(tool_call["args"])
    ...
```

**AgentExecutor — 자동 실행, 레거시 방식 (01a-executor.py)**

LangGraph 도입 이전의 기존 에이전트 런타임. `create_tool_calling_agent`로 에이전트를 정의한 뒤 `AgentExecutor`로 감싸 실행한다. 프롬프트에 `{agent_scratchpad}` 슬롯을 직접 뚫어야 하고, 호출 형식이 `{"input": "..."}`이다.

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

prompt = ChatPromptTemplate.from_messages([
    ("system", "도시 정보를 정확히 알려줘."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "..."})
print(result["output"])
```

**create_agent — 자동 실행, 최신 권장 (01b-agent.py)**

내부적으로 LangGraph `StateGraph`를 생성하는 고수준 래퍼. 에이전트 정의와 실행이 한 호출로 통합되고, 프롬프트의 scratchpad도 숨겨진다. 호출 형식은 `{"messages": [...]}`이고 결과는 `result["messages"]`에 들어있다 (pydantic ai와 동일).

```python
agent = create_agent(
    "gpt-4o",
    tools=[web_search, format_result],
    system_prompt="도시 정보를 정확히 알려줘.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
print(result["messages"][-1].content)
```

**세 방식 비교**

| | bind_tools | AgentExecutor | create_agent |
|---|---|---|---|
| 도구 실행 | 수동 (직접 루프) | 자동 (자체 런타임) | 자동 (LangGraph 런타임) |
| 호출 형식 | `chain.invoke(...)` | `{"input": "..."}` | `{"messages": [...]}` |
| 결과 접근 | `response.tool_calls` 등 | `result["output"]` | `result["messages"][-1].content` |
| 프롬프트 | 자유 | `{agent_scratchpad}` 필수 | 숨겨짐 |
| 상태 관리 | 직접 | `ConversationBufferMemory` 등 | `checkpointer` |
| HITL / interrupt | X | X | O |
| 상태 | 저수준 도구 | 레거시 (여전히 동작) | 최신 권장 |

**PydanticAI와 비교**

| | PydanticAI | bind_tools | AgentExecutor | create_agent |
|---|---|---|---|---|
| 도구 등록 | `@agent.tool` | `@tool` + `bind_tools()` | `@tool` + `AgentExecutor(...)` | `@tool` + `create_agent(tools=...)` |
| 도구 실행 | 자동 | 수동 | 자동 | 자동 |
| deps 접근 | `ctx.deps.building` | 클로저/전역변수 | 클로저/전역변수 | 클로저/전역변수 |
| 도구 호출 강제 | 없음 | `tool_choice="any"` 가능 | 없음 | 없음 |

### 3-2. Capability / 내장 도구

[03-tool/02-capability.py](../../langchain/03-tool/02-capability.py)

PydanticAI의 Capability는 LLM 네이티브 기능을 우선 사용하고 자동 우회할 수 있다.
```python
# PydanticAI
capabilities=[WebSearch(), WebFetch(allowed_domains=['aladin.co.kr'])]
```

LangChain에는 Capability와 동일한 기능은 존재하지 않지만, 커스텀 Tool을 구현하거나, type을 이용해서 llm native 기능을 사용하거나, 프레임워크에서 제공하는 Tool을 사용하는 방식으로 검색을 추가할 수 있다.

다음은 검색을 수행하는 커스텀 Tool을 구현한 것이다.<br>
pydantic ai에서의 allowed_domains는 자체적으로 구현해야 한다. 

```python
# LangChain — 도구 개별 생성 + 도메인 제한 직접 구현
@tool
def web_fetch(url: str) -> str:
    if "aladin.co.kr" not in url:
        return "허용되지 않는 도메인입니다."
    docs = WebBaseLoader(url).load()
    ...
```

type을 이용해서 llm의 native 기능을 사용하도록 요청할 수도 있다.

```python
# LangChain — OpenAI 네이티브 웹 검색 사용
llm_with_tools = init_chat_model("gpt-4o", model_provider="openai").bind_tools(
    [{"type": "web_search_preview"}]
)
```

또는 프레임워크에서 제공하는 Tool을 사용할 수도 있다.

```python
# LangChain — 프레임워크 제공 검색 도구
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
llm_with_tools = init_chat_model("gpt-4o", model_provider="openai").bind_tools([search])
```

**PydanticAI와 비교**

| | PydanticAI | LangChain |
|---|---|---|
| 웹 검색 | `WebSearch()` (1줄) | `{"type": "web_search_preview"}` 또는 `DuckDuckGoSearchRun()` |
| 웹 페이지 가져오기 | `WebFetch(allowed_domains=[...])` (1줄) | `@tool` + `WebBaseLoader` 직접 구현 |
| 도메인 제한 | `allowed_domains` 파라미터 | `if "도메인" not in url:` 직접 체크 |
| 네이티브 → 우회 자동 전환 | O | X |

**전체 코드**

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def web_fetch(url: str) -> str:
    """URL의 웹 페이지 내용을 가져옵니다. aladin.co.kr 도메인만 허용됩니다."""
    print("알라딘 페이지 읽는 중...")
    if "aladin.co.kr" not in url:
        return "허용되지 않는 도메인입니다. aladin.co.kr만 접근 가능합니다."
    docs = WebBaseLoader(url).load()
    text = "\n".join(doc.page_content for doc in docs)
    return text[:10000]

agent = create_agent(
    init_chat_model("gpt-4o", model_provider="openai"),
    tools=[{"type": "web_search_preview"}, web_fetch],
)

def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "오늘 삼성전자 주가를 조회해주고, aladin.co.kr에서 프로젝트 헤일메리 책 페이지를 찾아서 책 소개 부분을 요약해줘"}]}
    )
    content = result["messages"][-1].content
    if isinstance(content, list):
        print(content[0]["text"])
    else:
        print(content)

if __name__ == "__main__":
    main()
```

---

## 4. 워크플로우 (멀티에이전트)

여러 에이전트를 조합하여 복잡한 작업을 수행하는 대표적인 패턴으로 위임(Delegation)과 핸드오프(Handoff)가 있다. 이 외에도 병렬 실행, 투표/합의, Supervisor 패턴 등 다양한 조합 방식이 있지만, 여기서는 PydanticAI 예제와 대응되는 두 가지만 다룬다.<br>
PydanticAI와 패턴 자체는 동일하지만, 상태 공유 방식에서 차이가 있다.

### 4-1. 위임 (Delegation)

[04-workflow/01-delegation.py](../../langchain/04-workflow/01-delegation.py)

**하위 에이전트 정의**

먼저 위임받을 하위 에이전트를 생성한다.

```python
review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 가장 최신에 작성된 조건에 해당하는 리뷰를 선택해줘.",
)
```

**도구 안에서 하위 에이전트 호출**

`@tool` 안에서 `review_agent.invoke()`를 호출하면 위임이 된다.

```python
@tool
def review_movie(movie_title: str) -> str:
    """제목이 movie_title인 영화의 리뷰 찾기"""
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰를 찾아줘"}]}
    )
    return result["messages"][-1].content
```

**상위 에이전트에 도구 등록**

상위 에이전트가 이 도구를 자동으로 호출하면 하위 에이전트에 위임된다.

```python
movie_agent = create_agent("openai:gpt-4o", tools=[review_movie], ...)
```

**PydanticAI와 비교**

| | PydanticAI | LangChain |
|---|---|---|
| 도구 안 에이전트 호출 | `await review_agent.run(..., deps=ctx.deps)` | `review_agent.invoke({"messages": [...]})` |
| 상태 공유 | `deps=ctx.deps`로 전달 (타입 안전) | 없음 (클로저/전역변수) |

**전체 코드**

```python
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 가장 최신에 작성된 조건에 해당하는 리뷰를 선택해줘.",
)

@tool
def review_movie(movie_title: str) -> str:
    """제목이 movie_title인 영화의 리뷰 찾기"""
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰를 찾아줘"}]}
    )
    return result["messages"][-1].content

movie_agent = create_agent(
    "openai:gpt-4o",
    tools=[review_movie],
    system_prompt="영화 전문가들을 위한 영화를 선택해줘",
)

def main():
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": "2020년에 나온 한국 영화를 추천해 주고 그 영화의 리뷰를 알려줘"}]}
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
```

### 4-2. 핸드오프 (Handoff)

[04-workflow/02-handoff.py](../../langchain/04-workflow/02-handoff.py)

`main()`에서 에이전트를 순차 호출하고, 결과를 추출하여 다음 에이전트에 전달한다.

**구조화된 출력으로 결과 추출**

`response_format=MovieOutput`으로 구조화된 출력을 받아 `title`을 꺼낸다.

```python
movie_result = movie_agent.invoke({"messages": [...]})
movie = movie_result["structured_response"]  # MovieOutput 객체
```

**분기 처리는 직접**

에이전트를 써도 None 체크 같은 분기는 개발자가 처리해야 한다.

```python
if movie.title is None:
    print("영화를 찾지 못했습니다.")
    return
```

**다음 에이전트에 수동 연결**

```python
review_result = review_agent.invoke(
    {"messages": [{"role": "user", "content": f"제목이 {movie.title}인 영화의 리뷰를 찾아줘"}]}
)
```

**PydanticAI와 비교**

위임과 핸드오프 패턴 자체는 양쪽 모두 동일하다. 차이는 상태 공유(`deps` vs 전역변수)뿐이다.

| | 위임 (Delegation) | 핸드오프 (Handoff) |
|---|---|---|
| 제어 주체 | 상위 에이전트가 도구로 하위 에이전트 호출 | main()에서 순차 호출 |
| 상태 공유 | PydanticAI: ctx.deps / LangChain: 없음 | 불필요 (결과만 전달) |
| 에이전트 결합도 | 높음 (서로를 알고 있음) | 낮음 (서로의 존재를 모름) |

### 4-3. 체이닝을 이용한 핸드오프 구현

[04-workflow/02a-handoff-chain.py](../../langchain/04-workflow/02a-handoff-chain.py)

LCEL 파이프만으로도 어느 정도 워크플로우를 구현할 수 있지만 플로우가 복잡해질 수록 코드를 이해하고 유지보수하기가 어려워진다.

핸드오프를 파이프(`|`)로 완전히 연결할 수도 있다.

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

chain = (
    # [1] 영화 추천 → MovieOutput
    ChatPromptTemplate.from_messages([
        ("system", "영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘"),
        ("human", "{question}"),
    ])
    | init_chat_model("gpt-4o", model_provider="openai").with_structured_output(MovieOutput)
    # [2] 분기: title이 None이면 에러 메시지, 아니면 review_chain으로 파이프
    | RunnableBranch(
        (lambda movie: movie.title is None,
         RunnableLambda(lambda _: "영화를 찾지 못했습니다.")),
        # [3] MovieOutput → {"question": ...} → 리뷰 텍스트
        RunnableLambda(lambda movie: {"question": f"제목이 {movie.title}인 영화의 리뷰를 찾아줘"})
        | ChatPromptTemplate.from_messages([
            ("system", "영화 리뷰 전문가야. 가장 최신에 작성된 리뷰를 선택해줘."),
            ("human", "{question}"),
        ])
        | init_chat_model("gpt-4o", model_provider="openai")
        | StrOutputParser(),
    )
)

result = chain.invoke({"question": "2020년에 나온 한국 영화를 추천해 줘"})
```

전부 연결되어 "LCEL다운" 모양은 되지만, 다음과 같은 비용이 따른다:

- **분기 로직이 파이프 안으로 들어감** — `if/else`가 `RunnableBranch`와 lambda 조합으로 표현되어 직관성이 떨어짐
- **입력 스키마 변환도 lambda로 명시** — `MovieOutput` → `{"question": ...}` 변환이 파이프 중간에 끼어듦
- **에러 메시지/흐름 제어가 복잡** — "영화를 못 찾음" 케이스도 `RunnableLambda(lambda _: ...)` 래핑 필요
- **디버깅 어려움** — 여러 개의 lambda가 이어져 있으면 어느 단계에서 문제가 발생했는지 추적하기 힘듦

**복잡도 비교 (4-2 vs 4-3)**

| | 4-2 (순차 호출) | 4-3 (파이프 연결) |
|---|---|---|
| main() 길이 | ~9줄 | ~20줄 |
| 분기 표현 | `if/else` (명시적) | `RunnableBranch` + lambda 중첩 |
| 중간 타입 추적 | 변수 바인딩으로 명확 | 파이프 내부 타입 추론 필요 |
| 디버깅 | `print` 삽입이 쉬움 | `RunnableLambda`로 감싸야 중간값 확인 가능 |
| 에러 위치 | 스택트레이스에 라인 그대로 | lambda/RunnableLambda 내부로 숨음 |

이 정도의 복잡도가 쌓이면 **그래프가 필요하다는 신호**다. 5장에서 `StateGraph`로 같은 흐름을 선언적으로 표현할 수 있음을 보게 된다.

---

## 5. 그래프

워크플로우가 복잡해지는 경우 LangGraph를 이용해서 복잡성을 낮출 수 있다.

### 5-1. 복잡한 워크플로우의 문제점

[05-graph/01-workflow.py](../../langchain/05-graph/01-workflow.py)

04의 핸드오프 패턴에 **반복과 분기**가 추가되면 `main()`이 급격히 비대해진다.

```
사용자 입력 → 영화 추천 → 리뷰 검색 → 평점 확인 → (5점 이하면 종료, 초과면 반복)
```

**전체 코드**

```python
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain.agents import create_agent
from rich.prompt import Prompt

load_dotenv()

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

class ReviewScore(BaseModel):
    value: int = Field(description="리뷰 점수", ge=1, le=10)

movie_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘",
    response_format=MovieOutput,
)

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 리뷰를 1개만 찾아줘.",
)

score_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="리뷰 점수는 1점에서 10점 사이 정수값으로 변환해서 줘",
    response_format=ReviewScore,
)

def find_movie(year: str) -> MovieOutput:
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": f"{year}년에 개봉한 영화를 추천해 줘"}]}
    )
    return result["structured_response"]

def review_movie(movie_title: str) -> str:
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰를 1개만 찾아줘"}]}
    )
    return result["messages"][-1].content

def review_score(movie_title: str) -> ReviewScore:
    result = score_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰 점수를 찾아줘"}]}
    )
    return result["structured_response"]

def main():
    while True:
        year = Prompt.ask("영화 개봉 연도를 입력하세요")

        movie = find_movie(year)
        if movie.title is None:
            print("영화를 찾지 못해 종료합니다.")
            break
        print(f"영화 제목: {movie.title}")

        review = review_movie(movie.title)
        print(review)

        score = review_score(movie.title)
        if score.value <= 5:
            print("평점이 5점 이하인 경우 종료합니다.")
            break
        print(f"리뷰 점수: {score.value}")

if __name__ == "__main__":
    main()
```

### 5-2. LangGraph로 해결

[05-graph/02-graph.py](../../langchain/05-graph/02-graph.py)

LangGraph의 `StateGraph`를 사용하면 위 문제가 해결된다.

**구현 방식 요약**

LangGraph는 네 가지 요소로 그래프를 구성한다:

| 요소 | 구현 방식 | 설명 |
|---|---|---|
| **상태** | `TypedDict` | 노드 간 공유되는 전역 상태 |
| **노드** | `add_node("이름", 함수)` | 상태를 받아 상태를 반환하는 함수 |
| **엣지** | `add_edge("A", "B")` | 노드 간 연결을 **문자열**로 선언 |
| **분기** | `add_conditional_edges("A", 조건함수)` | 조건 함수가 다음 노드 **이름(문자열)**을 반환 |

#### 1. 상태 = TypedDict로 전역 정의

모든 노드가 하나의 상태 객체를 공유한다.<br> 
pydantic-graph의 `dataclass`와 달리 `TypedDict`를 사용한다.

```python
class WorkflowState(TypedDict):
    year: str
    movie_title: str
    review: str
    score: int
```

`StateGraph(WorkflowState)`로 그래프를 생성하면, 이 상태가 모든 노드에 전달된다.

#### 2. 노드 = 상태를 받아 새로 갱신할 상태를 반환하는 함수

노드는 일반 함수로 구현한다. `state`를 인자로 받고, **변경할 필드만 dict로 반환**하면 기존 상태에 병합된다.

```python
def find_movie_node(state: WorkflowState) -> WorkflowState:  # -> dict와 동일
    # state에서 값 읽기
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": f"{state['year']}년에 개봉한 영화를 추천해 줘"}]}
    )
    movie = result["structured_response"]
    title = movie.title if movie.title else ""

    # 변경할 필드만 반환 → 기존 state에 병합됨
    return {"movie_title": title}
```

pydantic-graph에서는 노드가 `@dataclass` 클래스이고 `run()` 메서드를 구현하지만,
LangGraph에서는 **일반 함수**로 더 간단하게 정의한다.

#### 3. 분기 = 조건 함수가 문자열 반환

분기 조건도 일반 함수로 구현한다.<br> 
상태를 보고 **다음 노드의 이름(문자열)** 을 반환한다.

```python
def check_movie_found(state: WorkflowState) -> str:
    return "review_movie" if state["movie_title"] else END
```

pydantic-graph에서는 `return ReviewMovieNode()` 또는 `return End(...)`로 **타입**을 반환하지만,
LangGraph에서는 `"review_movie"` 또는 `END`로 **문자열**을 반환한다.

#### 4. 그래프 조립 = add_node + add_edge

노드를 등록하고, 엣지로 연결하고, `compile()`로 실행 가능한 그래프를 생성한다.

```python
graph_builder = StateGraph(WorkflowState)

# 노드 등록
graph_builder.add_node("input", input_node)
graph_builder.add_node("find_movie", find_movie_node)
graph_builder.add_node("review_movie", review_movie_node)
graph_builder.add_node("check_score", check_score_node)

# 엣지 연결
graph_builder.add_edge(START, "input")
graph_builder.add_edge("input", "find_movie")
graph_builder.add_conditional_edges("find_movie", check_movie_found)
graph_builder.add_edge("review_movie", "check_score")
graph_builder.add_conditional_edges("check_score", check_score_value)

# 컴파일
movie_graph = graph_builder.compile()
```

pydantic-graph에서는 엣지를 따로 선언할 필요 없이 리턴 타입이 곧 엣지이지만,
LangGraph에서는 `add_edge`/`add_conditional_edges`로 **명시적으로 선언**해야 한다.

**PydanticAI와 비교**

| 요소 | LangGraph | pydantic-graph |
|---|---|---|
| **상태** | `TypedDict` (전역, dict 반환으로 병합) | `dataclass` (`ctx.state`로 직접 수정) |
| **노드** | 일반 함수 | `@dataclass` 클래스 + `run()` 메서드 |
| **엣지** | `add_edge("A", "B")` (문자열, 명시적) | `run()`의 리턴 타입 (타입, 암묵적) |
| **분기** | `add_conditional_edges` + 조건 함수 | `run()` 안의 로직 + Union 리턴 타입 |
| **타입 안전성** | 낮음 (문자열 기반, 오타 시 런타임 에러) | 높음 (리턴 타입으로 컴파일 시 검증) |
| **시각화** | `graph.get_graph().draw_mermaid()` | `graph.mermaid_code()` |

**전체 코드**

```python
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from rich.prompt import Prompt

load_dotenv()

class WorkflowState(TypedDict):
    year: str
    movie_title: str
    review: str
    score: int

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

class ReviewScore(BaseModel):
    value: int = Field(description="리뷰 점수", ge=1, le=10)

movie_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘",
    response_format=MovieOutput,
)

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 리뷰를 1개만 찾아줘.",
)

score_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="리뷰 점수는 1점에서 10점 사이 정수값으로 변환해서 줘",
    response_format=ReviewScore,
)

def input_node(state: WorkflowState) -> WorkflowState:
    year = Prompt.ask("영화 개봉 연도를 입력하세요")
    return {"year": year}

def find_movie_node(state: WorkflowState) -> WorkflowState:
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": f"{state['year']}년에 개봉한 영화를 추천해 줘"}]}
    )
    movie = result["structured_response"]
    title = movie.title if movie.title else ""
    print(f"영화 제목: {title}" if title else "영화를 찾지 못했습니다.")
    return {"movie_title": title}

def review_movie_node(state: WorkflowState) -> WorkflowState:
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {state['movie_title']}인 영화의 리뷰를 1개만 찾아줘"}]}
    )
    review = result["messages"][-1].content
    print(review)
    return {"review": review}

def check_score_node(state: WorkflowState) -> WorkflowState:
    result = score_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {state['movie_title']}인 영화의 리뷰 점수를 찾아줘"}]}
    )
    score = result["structured_response"]
    print(f"리뷰 점수: {score.value}")
    return {"score": score.value}

def check_movie_found(state: WorkflowState) -> str:
    return "review_movie" if state["movie_title"] else END

def check_score_value(state: WorkflowState) -> str:
    return "input" if state["score"] > 5 else END

graph_builder = StateGraph(WorkflowState)

graph_builder.add_node("input", input_node)
graph_builder.add_node("find_movie", find_movie_node)
graph_builder.add_node("review_movie", review_movie_node)
graph_builder.add_node("check_score", check_score_node)

graph_builder.add_edge(START, "input")
graph_builder.add_edge("input", "find_movie")
graph_builder.add_conditional_edges("find_movie", check_movie_found)
graph_builder.add_edge("review_movie", "check_score")
graph_builder.add_conditional_edges("check_score", check_score_value)

movie_graph = graph_builder.compile()

def main():
    print(movie_graph.get_graph().draw_mermaid())
    print("---")

    # TypedDict의 모든 키를 초기에 제공해야 하며, 각 노드에서 필요한 값으로 덮어쓴다
    result = movie_graph.invoke({
        "year": "",
        "movie_title": "",
        "review": "",
        "score": 0,
    })
    print(result)

if __name__ == "__main__":
    main()
```

---

## 6. RAG

[06-rag/01-rag.py](../../langchain/06-rag/01-rag.py)

RAG는 LangChain이 가장 강한 영역이다. PDF 로드, 청킹, 임베딩, 벡터 DB, 검색 체인이 모두 내장되어 있다.

**PDF 텍스트 변환 → 청킹 → 벡터 DB 저장**

PDF 로드, 청킹, 벡터 DB 저장이 각각 한 줄의 코드로 처리된다.

```python
docs = PyMuPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(...), persist_directory=CHROMA_PATH)
```

**RAG 체인 (파이프로 연결)**

`vectorstore.as_retriever()`로 검색기를 만들고 파이프에 연결한다.<br>
`search_kwargs={"k": 3}`은 가장 유사한 청크 3개를 반환하라는 의미다.

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 벡터 DB에서 검색한 문서(context)와 사용자 입력(question)을 조합해 프롬프트를 만들고 LLM에 전달
chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | init_chat_model("gpt-4o", model_provider="openai")
    | StrOutputParser()
)

response = chain.invoke("제주도의 해녀 문화에 대해 알려줘")
```

**PydanticAI와 비교**

| 단계 | PydanticAI | LangChain |
|---|---|---|
| PDF 텍스트 추출 | `fitz.open()` 직접 구현 | `PyMuPDFLoader(path).load()` (1줄) |
| 텍스트 청킹 | `split_text()` 직접 구현 | `RecursiveCharacterTextSplitter()` (1줄) |
| 임베딩 | `openai_client.embeddings.create()` 직접 호출 | `OpenAIEmbeddings()` (1줄) |
| 벡터 DB | `chromadb.PersistentClient()` 직접 관리 | `Chroma.from_documents()` (1줄) |
| 검색 | `collection.query()` 직접 구현 + `@agent.tool` 등록 | `retriever` 파이프 연결 |
| **전체 코드량** | **~128줄** | **~63줄** |
| **DB 교체** | 전부 다시 구현 | `Chroma` → `FAISS` 한 줄 교체 |

**전체 코드**

```python
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

PDF_PATH = os.path.join(os.path.dirname(__file__), "../../data/jeju_guide.pdf")
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "./chroma_db")

docs = PyMuPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"), persist_directory=CHROMA_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"PDF 로드 완료: {len(docs)}페이지, {len(chunks)}개 청크")

prompt = ChatPromptTemplate.from_messages([
    ("system", "검색된 문서를 기반으로 정확하게 답변해줘. 문서에 없는 내용은 추측하지 마.\n\n{context}"),
    ("human", "{question}"),
])

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | init_chat_model("gpt-4o", model_provider="openai")
    | StrOutputParser()
)

def main():
    response = chain.invoke("제주도의 해녀 문화에 대해 알려줘")
    print(response)

if __name__ == "__main__":
    main()
```

---

## 7.평가 : 비일관성과 강력함의 혼재

LangChain은 같은 목적을 달성하는 방법이 여러 개씩 공존한다.<br> 
LCEL → Agent Executor → LangGraph로 진화하면서 **과거 API를 deprecate하지 않고 유지** 해왔기 때문에 추상화가 비일관적이고 러닝 커브가 상승한다.

- 경로에 따라 **식별자·저장소·결과 접근 방식이 전부 달라짐**
- 튜토리얼/기존 코드에 세대가 섞여 있어 **어느 스타일을 보고 있는지 매번 구분**해야 함
- 방식에 따라 사용할 수 있는 기능 조합이 제한됨

이 장은 목적별로 공존하는 방법들을 한눈에 비교하는 카탈로그다.

### 7-1. 구조화된 출력

LLM 응답을 구조화된 객체로 받는 방법이 네 가지 공존한다.

| 방법 | API | 반환 타입 | 메커니즘 | 실행 시점 |
|---|---|---|---|---|
| `with_structured_output` | `llm.with_structured_output(CityInfo)` | Pydantic 인스턴스 | tool calling (API 강제) | 매 LLM 호출마다 |
| `JsonOutputParser` | `chain \| JsonOutputParser(pydantic_object=CityInfo)` | dict | 프롬프트 지시 (무시 가능) | 매 LLM 호출마다 |
| `PydanticOutputParser` | `chain \| PydanticOutputParser(pydantic_object=CityInfo)` | Pydantic 인스턴스 | 프롬프트 지시 (무시 가능) | 매 LLM 호출마다 |
| `response_format` | `create_agent(response_format=CityInfo)` | `result["structured_response"]` | tool calling | 에이전트 루프 종료 시 |

**비일관성 포인트**: 반환 타입도, 결과 접근 방식(`result` 자체 vs `result["structured_response"]` 등)도, 강제력(API vs 프롬프트)도 제각각. PydanticAI는 `output_type` 하나로 통일되어 있다.

### 7-2. 도구 등록

"검색 도구 하나 붙이기"에 최소 네 가지 경로가 있다.

| 경로 | 출처 | 실행 위치 | 코드 형태 |
|---|---|---|---|
| 직접 작성 + `bind_tools` | 개발자 | 로컬 + 수동 호출 루프 | `@tool def f(): ...` + `while response.tool_calls` |
| 직접 작성 + `create_agent` | 개발자 | 로컬 + 자동 루프 | `create_agent(tools=[f])` |
| 모델 네이티브 dict | 모델 제공자 | 제공자 서버 | `bind_tools([{"type": "web_search_preview"}])` |
| 프레임워크 제공 | `langchain_community.tools` | 로컬 | `DuckDuckGoSearchRun()` |

**비일관성 포인트**<br> 
tool을 위한 통일된 인터페이스가 없다.<br> 
네이티브 dict는 제공자 종속적(OpenAI면 `web_search_preview`, Anthropic이면 다른 문자열)이고, 일반 `@tool`과 함께 섞어 쓸 때는 어느 도구가 로컬 실행인지 개발자가 구분해야 한다.

### 7-3. 히스토리 관리

멀티 턴 대화 유지에 세 가지 축 × 두 가지 인프라 = 6가지 구현이 가능하다.

| | 메모리 저장소 | DB 저장소 | 식별자 | 래퍼 |
|---|---|---|---|---|
| 체인 (Runnable 계열) | `InMemoryChatMessageHistory` | `SQLChatMessageHistory` | `session_id` | `RunnableWithMessageHistory` |
| 에이전트 (LangGraph 계열) | `InMemorySaver` | `SqliteSaver` | `thread_id` | `create_agent(checkpointer=...)` |
| 그래프 (LangGraph 계열) | `InMemorySaver` | `SqliteSaver` | `thread_id` | `StateGraph.compile(checkpointer=...)` |

**비일관성 포인트** <br> 
Runnable 계열과 LangGraph 계열이 독립적으로 진화해서 **동일 개념에 이름·인터페이스가 다르게 붙어 있다**.<br> 
에이전트와 그래프는 `create_agent`가 LangGraph의 래퍼라서 인터페이스가 동일하지만, 체인은 완전히 다른 계열이다.

### 7-4. 에이전트 생성

과거 API가 deprecate되지 않고 남아 있어, 한 코드베이스에 여러 세대가 섞일 수 있다.

| 방법 | 상태 | 비고 |
|---|---|---|
| `create_agent` | 최신 권장 | 내부적으로 LangGraph |
| `StateGraph` 직접 작성 | 저수준 권장 | 그래프 모델 그대로 사용 |
| `AgentExecutor` + `create_tool_calling_agent` / `create_react_agent` | 레거시 (여전히 동작) | 튜토리얼에 여전히 등장 |
| `langgraph.prebuilt.create_react_agent` | 신세대 ReAct | 위와 **이름은 같지만 다른 구현** |
| `initialize_agent` | Deprecated | 여전히 import 가능 |

**비일관성 포인트** <br>
같은 이름(`create_react_agent`)이 **다른 모듈에서 다른 구현**으로 존재한다.<br>
튜토리얼을 따라하다가 어느 세대 코드를 보고 있는지 매번 확인해야 한다.

### 7-5. 패키지 분할

한 프레임워크의 기능이 6개 이상 패키지로 흩어져 있다.

| 기능 | 주 패키지 |
|---|---|
| 코어 추상 (`Runnable`, `ChatPromptTemplate`, `StrOutputParser`) | `langchain_core` |
| 주요 래퍼 (`create_agent`, `init_chat_model`) | `langchain` |
| 모델 통합 | `langchain_openai`, `langchain_anthropic`, `langchain_google_genai`, ... |
| 벡터 DB 통합 | `langchain_chroma`, `langchain_pinecone`, ... |
| 로더·일부 도구·레거시 | `langchain_community` |
| 그래프·체크포인터 | `langgraph`, `langgraph-checkpoint-sqlite`, `langgraph-checkpoint-postgres`, ... |

**비일관성 포인트** <br> 
같은 클래스가 버전 간 패키지를 옮겨 다닌다(예: 임베딩이 `langchain.embeddings` → `langchain_community.embeddings` → `langchain_openai`).<br>
예제 코드의 `import` 경로가 실제 패키지와 어긋나 `ImportError`를 만나는 일이 흔하다.

### 7-6. LangGraph의 고급 실행 제어

[07-tradeoff/01-graph-resume.py](../../langchain/07-tradeoff/01-graph-resume.py) — 중단·재개  
[07-tradeoff/02-time-travel.py](../../langchain/07-tradeoff/02-time-travel.py) — 시간여행

비일관성의 대가로 얻는 것은 **체인·AgentExecutor에 없는 고급 실행 제어**다. LangGraph 계열을 택했을 때 네 가지 기능이 추가로 열린다.

#### 상태 영속화 (checkpointer)

그래프의 **상태 + 실행 위치**가 checkpointer에 자동 저장된다. 같은 `thread_id`로 invoke하면 이전 지점에서 이어서 실행된다.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

graph = builder.compile(
    checkpointer=SqliteSaver(sqlite3.connect("workflow.db", check_same_thread=False))
)
config = {"configurable": {"thread_id": "user-1"}}
graph.invoke(input_dict, config=config)   # 다음 호출에서 이어서 실행 가능
```

체인의 `ChatMessageHistory`가 "메시지만" 저장한다면, LangGraph의 checkpointer는 "그래프 실행 전체 상태"를 저장한다.

#### 실행 중단·재개 (interrupt + Command)

노드 안에서 `interrupt()`를 호출하면 그래프가 멈추고 결과에 `__interrupt__`가 실린다. 다음 invoke에서 `Command(resume=값)`을 넘기면 그 값이 interrupt 자리로 들어오며 이어서 실행된다.

```python
from langgraph.types import Command, interrupt

def approval_node(state):
    answer = interrupt(f"'{state['movie']}' 리뷰를 찾을까요?")
    return {"approved": answer == "yes"}

# 1차: interrupt에서 정지
first = graph.invoke(input_dict, config=config)
print(first["__interrupt__"])   # → [Interrupt(value='...', ...)]

# 2차: Command(resume=...)로 재개 (다른 프로세스여도 됨)
final = graph.invoke(Command(resume="yes"), config=config)
```

두 invoke 사이에 **프로세스가 종료되어도 무관**하다. 상태가 checkpointer에 남아 있으므로 CLI 재시작·분산 워커 간 재개가 가능하다.

#### HITL (Human-in-the-loop)

위 `interrupt()`가 HITL의 기본 프리미티브다. "도구 호출 전 사람 승인", "위험한 작업 전 확인" 같은 패턴을 **그래프 문법에 자연스럽게 녹일 수 있다**. AgentExecutor나 체인에는 이런 프리미티브가 없어 콜백으로 우회 구현해야 한다.

```python
# 예: 이메일 발송 도구에 사람 승인을 끼워 넣기
def send_email_node(state):
    approval = interrupt({
        "action": "send_email",
        "to": state["recipient"],
        "body": state["draft"],
    })
    if approval["action"] != "approve":
        return {"sent": False, "reason": "rejected"}
    actually_send(state["recipient"], state["draft"])
    return {"sent": True}
```

#### 시간여행 (time travel)

체크포인트 이력을 조회하고, 과거 시점 상태를 편집한 뒤 그 시점부터 다시 실행할 수 있다.

```python
# 체크포인트 이력 조회 (최신순)
history = list(graph.get_state_history(config))
for snapshot in history:
    print(snapshot.values, snapshot.next)

# 특정 과거 시점 찾기 (예: dbl 노드 직전)
target = next(s for s in history if s.next == ("dbl",))

# 그 시점 상태를 편집 → 새 체크포인트 생성
new_config = graph.update_state(target.config, {"counter": 100})

# 편집된 시점부터 재실행 (None = 기존 상태로 계속)
replayed = graph.invoke(None, config=new_config)
```

디버깅·실험(다른 분기 탐색)에 유용하다. "만약 이 단계에서 다른 값이었다면 결과가 어떻게 달라졌을까"를 **실제로 돌려볼 수 있다**.

---

**정리**

| 기능 | 체인 / AgentExecutor | LangGraph |
|---|---|---|
| 상태 영속화 | 메시지 히스토리만 (`ChatMessageHistory`) | 전체 상태 + 실행 위치 (`checkpointer`) |
| 실행 중단·재개 | ✗ | ✓ (프로세스 재시작 간에도 유지) |
| HITL | 콜백·커스텀 래퍼로 우회 | ✓ (1급 프리미티브: `interrupt`) |
| 시간여행 | ✗ | ✓ (과거 편집·재실행) |

**비일관성과 강력함의 혼재** — LangGraph 계열로 넘어가면 API가 달라지는 비용을 치러야 하지만, 그 대가로 **다른 프레임워크에 없는 실행 제어 도구**를 얻는다. 에이전트가 단순 프롬프트-응답을 넘어 **장시간·중단가능·사람 개입**이 있는 실무 워크플로우로 확장될 때, 이 기능들이 있고 없고의 차이가 프레임워크 선택을 가른다.

### 7-7. 현재의 상태(AI 의견)

**그럼 지금은 많이 안정화되었는가**

부분적으로는 안정화되었지만 완전히 자리잡았다고 보긴 어렵다. 영역별로 차이가 크다.

**안정화된 영역**

- **코어 추상화**: `Runnable` 프로토콜, LCEL 파이프(`|`), `ChatPromptTemplate`, `StrOutputParser`, `init_chat_model`, `with_structured_output` — 최근 1~2년간 거의 변하지 않음
- **LangGraph 런타임**: `StateGraph`, checkpointer 인터페이스 — breaking change가 적고 프로덕션 사례 축적

**여전히 움직이는 영역**

- **에이전트 API**: `create_agent`는 비교적 최근에 `langchain.agents`로 통합되어 세부 변화가 있음. 구세대 `AgentExecutor`/`initialize_agent`가 deprecate되지 않아 튜토리얼과 서드파티 코드에 섞임. `create_react_agent`가 두 모듈에 다른 구현으로 존재
- **패키지 생태계**: 모델·벡터 DB 통합 패키지가 계속 늘어나고 경로가 이동(`langchain_community` → `langchain_openai` 등)
- **메모리/상태 관리**: `ConversationBufferMemory` → `RunnableWithMessageHistory` → LangGraph checkpointer로의 이주가 진행 중

| | 2023 말 | 현재 |
|---|---|---|
| 코어 API 변경 빈도 | 매주 | 수개월에 한 번 |
| LCEL | 새 기능, 혼란 | 표준 |
| 에이전트 API | 격변 중 | `create_agent`로 수렴 중이나 레거시 혼재 |
| 패키지 구조 | 모놀리식 | 분할 완료, 계속 늘어남 |
| 프로덕션 채택 | 실험적 | 실사용 보편화 |

**결론**: "초기의 급격한 변화는 지나갔다" 수준에서 "Django/Rails만큼 안정적" 사이 어딘가다.

- **코어(LCEL + LangGraph)만 쓰는 새 프로젝트**라면 이제 안전하게 써도 된다
- **튜토리얼을 따라하거나 레거시 코드를 관리**해야 한다면 여전히 세대 구분 비용이 남아 있다 — 이 장의 비일관성이 그 비용이다

쓰기 시작할 때의 고민은 줄었지만, 생태계를 돌아다니며 예제를 참고할 때의 고민은 여전히 존재한다.

**권장**: 
새 코드라면 **LangGraph 계열(`create_agent` 또는 `StateGraph`) + 최신 패키지 경로**로 통일한다. 유연성의 대가를 줄이는 가장 현실적인 방법이다.

