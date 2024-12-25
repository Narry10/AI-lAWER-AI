from fastapi import FastAPI, HTTPException
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from services.GenerationAnswer import GenerationAnswer

load_dotenv("./env/.env")

HF_TOKEN = os.getenv("HF_TOKEN")

login(HF_TOKEN)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qa_model = GenerationAnswer()

class QuestionAnswer(BaseModel):
    description: str = Field(..., min_length=1, description="Mô tả câu hỏi không được để trống")
    specificSituation: str = Field(..., min_length=1, description="Tình huống cụ thể không được để trống")
@app.get("/")
async def root():
    return RedirectResponse(url="/docs")


@app.post("/get_answer")
async def get_answer(question: QuestionAnswer):
    combined_question = f"{question.description} {question.specificSituation}"

    try:
        answer = qa_model.get_answer(combined_question)

        if answer is None or answer.strip() == "":
            raise HTTPException(status_code=404, detail="Không tìm thấy câu trả lời")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Lỗi xác thực dữ liệu: {str(ve)}")
    except ConnectionError:
        raise HTTPException(status_code=503, detail="Không thể kết nối tới dịch vụ mô hình")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    return {"question": combined_question, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
