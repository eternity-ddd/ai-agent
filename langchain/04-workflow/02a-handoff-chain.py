"""
핸드오프(Handoff) 패턴: 파이프로 억지 연결한 버전

- movie_chain | RunnableLambda | review_chain 형태로 파이프를 연결
- 중간에 None 체크, 필드 추출, 프롬프트 재조립을 RunnableLambda 안에서 처리
- 파이프로 연결은 되지만, lambda 안에 비즈니스 로직이 들어가서 가독성이 떨어짐
- 02-handoff.py의 create_agent 버전과 비교하면 파이프의 한계가 드러남

대응하는 pydantic 예제: pydantic/04-workflow/02-handoff.py
"""
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# --- 모델 정의 ---

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

# --- 체인 정의 ---

movie_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘"),
        ("human", "{question}"),
    ])
    | init_chat_model("gpt-4o", model_provider="openai").with_structured_output(MovieOutput)
)

review_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "영화 리뷰 전문가야. 가장 최신에 작성된 리뷰를 선택해줘."),
        ("human", "{question}"),
    ])
    | init_chat_model("gpt-4o", model_provider="openai")
    | StrOutputParser()
)

# --- 파이프로 억지 연결: RunnableLambda 안에 분기/변환 로직이 들어감 ---

chain = (
    movie_chain
    | RunnableLambda(lambda movie:
        review_chain.invoke({"question": f"제목이 {movie.title}인 영화의 리뷰를 찾아줘"})
        if movie.title is not None
        else "영화를 찾지 못했습니다."
    )
)

# --- 실행 ---

def main():
    result = chain.invoke({"question": "2020년에 나온 한국 영화를 추천해 줘"})
    print(result)

if __name__ == "__main__":
    main()
