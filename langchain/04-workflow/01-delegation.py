"""
위임(Delegation) 패턴: 에이전트가 도구를 통해 다른 에이전트를 호출

- movie_agent가 영화를 선택한 뒤, @tool 내부에서 review_agent를 호출하여 리뷰를 검색
- create_agent는 도구 호출 루프를 자동으로 처리
- PydanticAI의 @agent.tool 위임 패턴과 동일한 구조

대응하는 pydantic 예제: pydantic/04-workflow/01-delegation.py
"""
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# --- 하위 에이전트 (리뷰 검색) ---

review_agent = create_agent(
    "openai:gpt-4o",
    system_prompt="영화 리뷰 전문가야. 가장 최신에 작성된 조건에 해당하는 리뷰를 선택해줘.",
)

# --- 도구 정의 (movie_agent가 review_agent에 위임) ---

@tool
def review_movie(movie_title: str) -> str:
    """제목이 movie_title인 영화의 리뷰 찾기"""
    result = review_agent.invoke(
        {"messages": [{"role": "user", "content": f"제목이 {movie_title}인 영화의 리뷰를 찾아줘"}]}
    )
    return result["messages"][-1].content

# --- 상위 에이전트 (영화 선택 + 리뷰 위임) ---

movie_agent = create_agent(
    "openai:gpt-4o",
    tools=[review_movie],
    system_prompt="영화 전문가들을 위한 영화를 선택해줘",
)

# --- 실행 ---

def main():
    result = movie_agent.invoke(
        {"messages": [{"role": "user", "content": "2020년에 나온 한국 영화를 추천해 주고 그 영화의 리뷰를 알려줘"}]}
    )
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
