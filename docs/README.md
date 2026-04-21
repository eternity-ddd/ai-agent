# PydanticAI vs LangChain 비교 자료

## 배경: 에이전트란 무엇인가

### 자율적인 에이전트

AI 에이전트는 **LLM이 스스로 판단하여 도구를 선택하고 실행하는 시스템**이다.
사용자가 "서울의 날씨를 알려줘"라고 하면, 에이전트가 알아서 검색 도구를 호출하고 결과를 정리한다.

```
사용자 → LLM(판단) → 도구 선택 → 도구 실행 → 결과 → LLM(정리) → 응답
```

이것이 가능한 이유는 **Function Calling** 덕분이다.

### ReAct 에이전트

자율적 에이전트의 대표 패턴. LLM이 **Thought(생각) → Action(도구 호출) → Observation(관찰) → Thought** 루프를 돌며 답을 만든다.

```
사용자 질문
  → [Thought] "검색이 필요해"
  → [Action]  web_search("서울 날씨")
  → [Observation] "맑음, 20도"
  → [Thought] "충분한 정보, 답변 가능"
  → 최종 응답
```

오늘날 Function Calling 기반 에이전트의 사실상 표준 패턴으로, PydanticAI의 `@agent.tool`과 LangChain의 `create_agent`가 모두 이 루프를 내부적으로 자동화한다.

### Function Calling: 에이전트의 기반 기술

Function Calling은 LLM이 **"이 함수를 이 인자로 호출해줘"**라고 구조화된 요청을 반환하는 기능이다.
LLM이 직접 함수를 실행하는 것이 아니라, **호출 의도만 표현**하고 실제 실행은 프레임워크가 처리한다.

```
1. 개발자가 도구의 스키마(이름, 파라미터, 설명)를 LLM에 전달
2. LLM이 사용자 질문을 분석하여 적절한 도구와 인자를 선택
3. LLM이 {"name": "web_search", "args": {"query": "서울 날씨"}} 형태로 반환
4. 프레임워크가 실제 함수를 실행하고 결과를 LLM에 다시 전달
5. LLM이 결과를 바탕으로 최종 응답 생성
```

PydanticAI와 LangChain은 모두 이 Function Calling 위에 구축되어 있다.
차이는 **이 과정을 얼마나 자동화하고, 어떤 방식으로 추상화하느냐**이다.

### 워크플로우 vs 자율적 에이전트

에이전트를 구현할 때 두 가지 접근이 있다:

**자율적 에이전트** — LLM이 다음 행동을 스스로 결정

```python
# LLM이 알아서 도구를 선택하고 실행
agent.run("서울에 대해 알려줘. 유명한 건물도 포함해줘.")
# → LLM이 판단: "검색이 필요하겠군" → web_search 호출 → "포맷팅도 하자" → format 호출
```

- 유연하지만 예측 불가능
- 도구를 안 쓸 수도, 엉뚱한 도구를 쓸 수도 있음
- PydanticAI의 `@agent.tool`, LangChain의 `create_agent`가 이 방식

**워크플로우** — 개발자가 흐름을 통제

```python
# 개발자가 단계를 명시적으로 정의
movie = find_movie(year)         # 1단계
if movie.title is None: break    # 개발자가 분기 결정
review = review_movie(movie)     # 2단계
score = review_score(movie)      # 3단계
```

- 예측 가능하고 디버깅 쉬움
- 유연성이 떨어짐
- 복잡해지면 main()이 비대해짐 → **그래프로 해결**

**그래프** — 워크플로우를 구조화

```python
# 단계를 노드로, 흐름을 엣지로 선언
graph.add_node("find_movie", find_movie_node)
graph.add_node("review", review_node)
graph.add_conditional_edges("find_movie", check_found)
```

- 워크플로우의 구조화된 버전
- 분기/반복/상태 관리가 명확
- pydantic-graph, LangGraph가 이 방식

실무에서는 **자율적 에이전트 + 워크플로우를 조합**한다:
각 노드 안에서는 에이전트가 자율적으로 동작하고, 노드 간 흐름은 개발자가 통제한다.

### 프레임워크에서의 "Agent"

에이전트의 핵심은 **"LLM이 스스로 판단하여 도구를 선택하고 호출하는 것"**이다.
Tool 없이 LLM만 호출하면 그냥 채팅이지 에이전트가 아니다.

프레임워크에서는 "Agent"라는 이름을 쓰더라도 도구가 없으면 실질적으로 채팅과 같다:

| | 의미 | 도구 | 자율성 |
|---|---|---|---|
| **일반적인 AI Agent** | LLM이 스스로 판단하여 행동하는 시스템 | 필수 | 높음 |
| **PydanticAI의 Agent** | LLM + 도구 + 상태를 캡슐화한 **실행 단위** | 선택 (없으면 채팅) | 도구가 있을 때 높음 |
| **LangChain의 create_agent** | LLM + 도구 + 도구 실행 루프를 자동화한 **LangGraph 노드** | 선택 (없으면 채팅) | 도구가 있을 때 높음 |
| **LangChain의 init_chat_model** | LLM을 직접 사용하는 **모델 래퍼** | 없음 | 낮음 (개발자가 모든 것을 제어) |

---

## 비교의 목적

AI 에이전트를 만들 때 PydanticAI와 LangChain은 가장 대표적인 두 가지 선택지이다.
둘 다 같은 일을 할 수 있지만 **설계 철학이 다르다**.

- **PydanticAI** (2024~): Agent 하나에 모든 것을 캡슐화. Python 타입 시스템 활용.
- **LangChain** (2022~): 구성 요소를 파이프로 조합. 풍부한 생태계와 인프라.

이 자료는 동일한 기능을 양쪽으로 구현하여 **코드 레벨에서 차이를 직접 비교**한다.

## 자료 구조

```
pydantic/          ← PydanticAI 예제 (13개 파일)
langchain/         ← LangChain 예제 (16개 파일)
docs/
  pydantic/        ← PydanticAI 문서 (단계별 설명)
  langchain/       ← LangChain 문서 (PydanticAI 비교 포함)
  README.md        ← 이 파일
data/              ← RAG용 PDF 데이터
```

## 목차 (양쪽 동일)

| # | 주제 | 핵심 비교 포인트 |
|---|---|---|
| 01 | 기본 (채팅, 싱글 턴(Single-turn), 멀티 턴(Multi-turn)) | Agent vs 파이프 체인 |
| 02 | 의존성과 출력 관리 | deps(타입 안전) vs 전역변수(문자열 키) |
| 03 | 도구 (Tool) | 자동 실행 vs 수동 루프, Capability vs 도구 직접 조합 |
| 04 | 워크플로우 (멀티에이전트) | 위임/핸드오프 패턴, 상태 공유 방식 |
| 05 | 그래프 | 리턴 타입 기반(타입 안전) vs 문자열 기반(유연) |
| 06 | RAG | 직접 구현(~128줄) vs 내장 컴포넌트(~63줄) |

## 종합 비교

| 영역 | PydanticAI | LangChain | 어디가 강한가 |
|---|---|---|---|
| 기본 사용법 | Agent 하나로 완결 | 파이프 조합 필요 | PydanticAI |
| 상태 관리 | deps (타입 안전) | 전역변수 (타입 불안전) | PydanticAI |
| 구조화된 출력 | output_type 하나 | 체인/에이전트 별도 방식 | PydanticAI |
| 도구 실행 | 자동 | 수동 or create_agent | 비슷 |
| Capability | 네이티브 자동 전환 | 직접 조합 | PydanticAI |
| 히스토리 관리 | 수동 전달 | 자동 누적 + 영속화 | LangChain |
| 그래프 | 타입 기반 (안전) | 문자열 기반 (유연) | 취향 |
| RAG | 직접 구현 (~128줄) | 내장 컴포넌트 (~63줄) | LangChain |
| 생태계/도구 | 적음 | 풍부 | LangChain |
| 학습 곡선 | 낮음 | 높음 | PydanticAI |
| 디버깅 | 직관적 | 추상화로 인해 어려움 | PydanticAI |

## 도구 선택 기준

**PydanticAI를 선택할 때:**
- 에이전트 코어에 집중하고 싶을 때
- 타입 안전성이 중요할 때
- 빠른 프로토타이핑이 필요할 때
- 코드의 가독성과 디버깅이 중요할 때

**LangChain을 선택할 때:**
- RAG, 벡터 DB, 히스토리 관리 등 인프라가 필요할 때
- 다양한 모델/도구를 조합해야 할 때
- 프로덕션 레벨의 체크포인팅, 영속화가 필요할 때
- 기존 LangChain 생태계와 통합해야 할 때

> **핵심**: PydanticAI는 **에이전트를 만드는 데 집중**하고, LangChain은 **에이전트 주변의 모든 것**을 제공한다.

## 실행 방법

```bash
# 의존성 설치
uv sync

# PydanticAI 예제 실행
uv run python pydantic/01-basic/01-chat.py

# LangChain 예제 실행
uv run python langchain/01-basic/01-chat.py

# RAG 예제 실행 전 PDF 생성
uv run python data/generate_pdf.py
uv run python pydantic/06-rag/01-rag.py
uv run python langchain/06-rag/01-rag.py
```

`.env` 파일에 API 키 설정이 필요하다:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
