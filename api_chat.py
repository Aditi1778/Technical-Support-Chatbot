import logging
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app_graph import graph, PREDEFINED_QUESTIONS, AgentState
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio
from uuid import uuid4
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="TechBit Chatbot API",
    description="A multilingual support chatbot for procedural queries using WhatsApp chat data and document manuals.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str | None = None
    predefined_index: int | None = None

class QuestionResponse(BaseModel):
    question: str
    answer: str

class PredefinedQuestionsResponse(BaseModel):
    questions: List[str]

# Endpoints
@app.get("/predefined-questions", response_model=PredefinedQuestionsResponse)
async def get_predefined_questions():
    """Retrieve the list of predefined questions."""
    logger.info("Fetching predefined questions")
    return {"questions": PREDEFINED_QUESTIONS}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question (custom or predefined by index) and get an answer."""
    logger.info(f"Received question request: {request}")
    
    # Determine the question
    if request.predefined_index is not None:
        if 0 <= request.predefined_index < len(PREDEFINED_QUESTIONS):
            question = PREDEFINED_QUESTIONS[request.predefined_index]
            logger.info(f"Selected predefined question: {question}")
        else:
            logger.error(f"Invalid predefined question index: {request.predefined_index}")
            raise HTTPException(status_code=400, detail=f"Predefined question index must be between 0 and {len(PREDEFINED_QUESTIONS)-1}")
    elif request.question:
        question = request.question.strip()
        logger.info(f"Received custom question: {question}")
    else:
        logger.error("No question or predefined index provided")
        raise HTTPException(status_code=400, detail="Must provide either a question or a predefined_index")

    if not question:
        logger.error("Empty question provided")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Initialize state
    state = {
        "messages": [HumanMessage(content=question)]
    }
    logger.info(f"Initial state: {state}")

    try:
        # Run the graph
        result = await graph.ainvoke(state)
        logger.info(f"Graph result: {result}")

        # Extract answer
        messages = result.get("messages", [])
        error = result.get("error", None)

        if error:
            logger.error(f"Graph returned error: {error}")
            raise HTTPException(status_code=500, detail=f"Error processing question: {error}")
        elif messages and len(messages) > 1 and isinstance(messages[-1], SystemMessage):
            answer = messages[-1].content
            logger.info(f"Generated answer: {answer}")
            return {"question": question, "answer": answer}
        else:
            logger.warning("No valid answer found in graph result")
            raise HTTPException(status_code=500, detail="No answer generated")
    except Exception as e:
        logger.error(f"Error running graph: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)