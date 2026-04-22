"""
핸드오프(Handoff) 패턴: 파이프로 완전히 연결한 버전

- movie_chain → RunnableBranch(분기) → RunnableLambda(입력 변환) → review_chain
- 분기·변환·실행이 전부 파이프 안으로 들어가 하나의 체인으로 연결됨
- "LCEL다운" 모양은 되지만 분기 로직이 파이프에 혼입되어 복잡도가 크게 올라감
- 02-handoff.py의 순차 호출 방식과 비교하면 파이프의 한계가 드러남

대응하는 pydantic 예제: pydantic/04-workflow/02-handoff.py
"""
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

# --- 모델 정의 ---

class MovieOutput(BaseModel):
    title: str | None = Field(description="영화 제목", default=None, max_length=100)

chain = (
    # [1] 영화 추천 → MovieOutput
    ChatPromptTemplate.from_messages([
        ("system", "영화 전문가들을 위한 영화를 선택해줘. 적절한 영화가 없으면 title을 null로 반환해줘"),
        ("human", "{question}"),
    ])
    | init_chat_model("gpt-4o", model_provider="openai").with_structured_output(MovieOutput)
    # [2] 분기: title이 None이면 에러 메시지, 아니면 review_chain으로 파이프
    | RunnableBranch(
        (lambda movie: movie.title is None, RunnableLambda(lambda _: "영화를 찾지 못했습니다.")),
        RunnableLambda(lambda movie: {"question": f"제목이 {movie.title}인 영화의 리뷰를 찾아줘"})
        # [3] MovieOutput → {"question": ...} → 리뷰 텍스트
        | ChatPromptTemplate.from_messages([
            ("system", "영화 리뷰 전문가야. 가장 최신에 작성된 리뷰를 선택해줘."),
            ("human", "{question}"),
        ])
        | init_chat_model("gpt-4o", model_provider="openai")
        | StrOutputParser(),
    )
)

# --- 실행 ---

def main():
    result = chain.invoke({"question": "2020년에 나온 한국 영화를 추천해 줘"})
    print(result)

if __name__ == "__main__":
    main()
