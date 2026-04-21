# LangChain

(+) **풍부한 built-in 컴포넌트** — RAG, 히스토리 관리, 벡터 DB 등 인프라성 기능 내장 ([6장](#6-rag))       
(+) **히스토리 자동 관리** — `RunnableWithMessageHistory` + `session_id`로 영속화까지 ([1-3](#1-3-멀티-턴multi-turn))       
(+) **LCEL 파이프(`|`)** — 구성 요소를 선언적으로 조합 ([1장](#1-기본))       
(+) **두 가지 실행 방식** — 모델 직접(`init_chat_model`) vs 에이전트(`create_agent`) 상황별 선택       
(-) **높은 추상화** — 상황별로 적합한 컴포넌트를 익혀야 하고 디버깅이 어려움       
(-) **타입 안전성 부족** — 상태 주입이 `{length}` 같은 문자열 키 기반, 오타 시 런타임 에러 ([2-1](#2-1-상태와-출력))       
(-) **방식의 이원화** — 체인(`with_structured_output`)과 에이전트(`response_format`)가 별도 방식 ([2-2a](#2-2a-에이전트에서-구조화된-출력))

## 목차

- [1. 기본](#1-기본)
  - [1-1. 채팅](#1-1-채팅)
  - [1-2. 싱글 턴(Single-turn)](#1-2-싱글-턴single-turn)
  - [1-3. 멀티 턴(Multi-turn)](#1-3-멀티-턴multi-turn)
- [2. 의존성과 출력 관리](#2-의존성과-출력-관리)
  - [2-1. 상태와 출력](#2-1-상태와-출력)
  - [2-2. JSON 출력](#2-2-json-출력)
  - [2-2a. 에이전트에서 구조화된 출력](#2-2a-에이전트에서-구조화된-출력)
  - [2-3. Pydantic Validation 체크](#2-3-pydantic-validation-체크)
- [3. Tool을 이용한 Agent 구현](#3-tool을-이용한-agent-구현)
  - [3-1. Tool 등록](#3-1-tool-등록)
  - [3-2. Capability / 내장 도구](#3-2-capability--내장-도구)
- [4. 워크플로우 (멀티에이전트)](#4-워크플로우-멀티에이전트)
  - [4-1. 위임 (Delegation)](#4-1-위임-delegation)
  - [4-2. 핸드오프 (Handoff)](#4-2-핸드오프-handoff)
- [5. 그래프](#5-그래프)
  - [5-1. 복잡한 워크플로우의 문제점](#5-1-복잡한-워크플로우의-문제점)
  - [5-2. LangGraph로 해결](#5-2-langgraph로-해결)
- [6. RAG](#6-rag)

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

llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0.3, max_tokens=500)

chain = prompt | llm | StrOutputParser()
```

파이프 없이 각 Runnable의 `invoke()`를 수동으로 호출해도 동일한 결과가 나온다:

```python
response = prompt.invoke({"question": "서울에 대해 알려줘"})
response = llm.invoke(response)
response = parser.invoke(response)
```

**실행**

```python
response = chain.invoke({"question": "서울에 대해 알려줘"})
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

llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0.3, max_tokens=500)

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
    MessagesPlaceholder("history"),   # ← 이전 대화가 여기에 자동 삽입
    ("human", "{question}"),
])
```

**RunnableWithMessageHistory로 자동 관리**

`session_id`로 여러 사용자의 대화를 분리 관리할 수 있다.

```python
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

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

---

## 2. 의존성과 출력 관리

### 2-1. 상태와 출력

[02-deps-and-output/01-deps-and-output.py](../../langchain/02-deps-and-output/01-deps-and-output.py)

**구조화된 출력**

`with_structured_output(CityInfo)`로 LLM 응답을 Pydantic BaseModel로 구조화한다.

```python
llm = init_chat_model("gpt-4o", model_provider="openai").with_structured_output(CityInfo)
chain = prompt | llm
```

**상태 관리**

LangChain에는 PydanticAI의 `deps`가 없어 전역 변수로 상태를 관리하고,
프롬프트 변수(`{length}`)로 전달해야 한다.

```python
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
- `get_format_instructions()`: LLM에게 "이 JSON 형식으로 답해"라고 **지시** (프롬프트에 삽입)
- 파이프 안에서: LLM 응답을 실제로 dict로 **파싱**

```python
parser = JsonOutputParser(pydantic_object=CityInfo)
chain = prompt | llm | parser
```

`with_structured_output()`은 API 레벨에서 JSON을 강제하지만,
`JsonOutputParser`는 프롬프트 텍스트로 유도하므로 LLM이 무시할 수 있다.

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

`create_agent`에서는 `response_format`으로 구조화된 출력을 지정한다.
체인의 `with_structured_output()`과는 별개의 방식이다.

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

[03-tool/01-tool.py](../../langchain/03-tool/01-tool.py) — 수동 실행, [03-tool/01a-agent.py](../../langchain/03-tool/01a-agent.py) — 자동 실행

LangChain에서 도구를 등록하는 두 가지 방식이 있다.

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

**create_agent — 자동 실행 (01a-agent.py)**

`create_agent`는 도구 호출 → 실행 → 결과 전달 → 재호출 루프를 자동으로 처리한다.

```python
agent = create_agent(
    "gpt-4o",
    tools=[web_search, format_result],
    system_prompt="도시 정보를 정확히 알려줘.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

**PydanticAI와 비교**

| | PydanticAI | LangChain (bind_tools) | LangChain (create_agent) |
|---|---|---|---|
| 도구 등록 | `@agent.tool` | `@tool` + `bind_tools()` | `@tool` + `create_agent(tools=...)` |
| 도구 실행 | 자동 | 수동 (직접 루프) | 자동 |
| deps 접근 | `ctx.deps.building` | 클로저/전역변수 | 클로저/전역변수 |
| 도구 호출 강제 | 없음 (LLM 자율) | `tool_choice="any"` 가능 | 없음 |

### 3-2. Capability / 내장 도구

[03-tool/02-capability.py](../../langchain/03-tool/02-capability.py)

PydanticAI의 Capability는 LLM 네이티브 기능을 우선 사용하고 자동 우회하지만,
LangChain에서는 각 도구를 개별 생성하고 도메인 제한도 직접 구현해야 한다.

```python
# PydanticAI — 한 줄
capabilities=[WebSearch(), WebFetch(allowed_domains=['aladin.co.kr'])]

# LangChain — 도구 개별 생성 + 도메인 제한 직접 구현
@tool
def web_fetch(url: str) -> str:
    if "aladin.co.kr" not in url:
        return "허용되지 않는 도메인입니다."
    docs = WebBaseLoader(url).load()
    ...
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
    "openai:gpt-4o",
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

여러 에이전트를 조합하여 복잡한 작업을 수행하는 두 가지 패턴이 있다.
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
`create_agent`를 써도 에이전트 간 연결은 수동이다 — 이것이 그래프가 필요한 이유.

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

**파이프로 억지 연결** ([02a-handoff-chain.py](../../langchain/04-workflow/02a-handoff-chain.py))

`RunnableLambda`로 파이프 연결은 가능하지만, lambda 안에 분기 로직이 들어가서 가독성이 떨어진다:

```python
chain = (
    movie_chain
    | RunnableLambda(lambda movie:
        review_chain.invoke({"question": f"제목이 {movie.title}인 영화의 리뷰를 찾아줘"})
        if movie.title is not None
        else "영화를 찾지 못했습니다."
    )
)
```

**PydanticAI와 비교**

위임과 핸드오프 패턴 자체는 양쪽 모두 동일하다. 차이는 상태 공유(`deps` vs 전역변수)뿐이다.

| | 위임 (Delegation) | 핸드오프 (Handoff) |
|---|---|---|
| 제어 주체 | 상위 에이전트가 도구로 하위 에이전트 호출 | main()에서 순차 호출 |
| 상태 공유 | PydanticAI: ctx.deps / LangChain: 없음 | 불필요 (결과만 전달) |
| 에이전트 결합도 | 높음 (서로를 알고 있음) | 낮음 (서로의 존재를 모름) |

---

## 5. 그래프

워크플로우가 복잡해지면 (분기, 반복, 조건부 종료), `create_agent`만으로는 한계가 있다.
LangChain에서는 **LangGraph의 StateGraph**로 이를 해결한다.

### 5-1. 복잡한 워크플로우의 문제점

[05-graph/01-workflow.py](../../langchain/05-graph/01-workflow.py)

04의 핸드오프 패턴에 **반복과 분기**가 추가되면 `main()`이 급격히 비대해진다.

```
사용자 입력 → 영화 추천 → 리뷰 검색 → 평점 확인 → (5점 이하면 종료, 초과면 반복)
```

`create_agent`를 써도 이 분기/반복 로직은 `main()`의 `if/while`에 흩어진다.

```python
def main():
    while True:                          # 반복
        movie = find_movie(year)
        if movie.title is None:          # 분기 1
            break
        review = review_movie(movie.title)
        score = review_score(movie.title)
        if score.value <= 5:             # 분기 2
            break
```

이 문제는 PydanticAI도 동일하다.

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
| **노드** | 함수 + `add_node("이름", 함수)` | 상태를 받아 상태를 반환하는 함수 |
| **엣지** | `add_edge("A", "B")` | 노드 간 연결을 **문자열**로 선언 |
| **분기** | `add_conditional_edges("A", 조건함수)` | 조건 함수가 다음 노드 **이름(문자열)**을 반환 |

**1. 상태 = TypedDict로 전역 정의**

모든 노드가 하나의 상태 객체를 공유한다. pydantic-graph의 `dataclass`와 달리 `TypedDict`를 사용한다.

```python
class WorkflowState(TypedDict):
    year: str
    movie_title: str
    review: str
    score: int
```

`StateGraph(WorkflowState)`로 그래프를 생성하면, 이 상태가 모든 노드에 전달된다.

**2. 노드 = 상태를 받아 상태를 반환하는 함수**

노드는 일반 함수로 구현한다. `state`를 인자로 받고, **변경할 필드만 dict로 반환**하면 기존 상태에 병합된다.

```python
def find_movie_node(state: WorkflowState) -> WorkflowState:
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

**3. 분기 = 조건 함수가 문자열 반환**

분기 조건도 일반 함수로 구현한다. 상태를 보고 **다음 노드의 이름(문자열)**을 반환한다.

```python
def check_movie_found(state: WorkflowState) -> str:
    return "review_movie" if state["movie_title"] else END
```

pydantic-graph에서는 `return ReviewMovieNode()` 또는 `return End(...)`로 **타입**을 반환하지만,
LangGraph에서는 `"review_movie"` 또는 `END`로 **문자열**을 반환한다.

**4. 그래프 조립 = add_node + add_edge**

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

**PDF → 청킹 → 벡터 DB (각 1줄)**

```python
docs = PyMuPDFLoader(PDF_PATH).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(...), persist_directory=CHROMA_PATH)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

**RAG 체인 (파이프로 연결)**

```python
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
