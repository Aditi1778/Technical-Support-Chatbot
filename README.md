
TechBit Chatbot Project
======================

Project Overview
---------------
This project is a multilingual support chatbot designed to provide procedural answers based on document manuals and WhatsApp chat data. It leverages LangChain, LangGraph, and FastAPI to process .docx documents, parse WhatsApp chat logs, create a vector store for retrieval, and serve responses via a REST API. The chatbot supports queries in English, Hindi, or Hinglish and provides step-by-step instructions for predefined or custom questions.

Features
--------
- Processes .docx manuals and WhatsApp chat data (CSV format).
- Uses Google Generative AI for embeddings and language model responses.
- Supports predefined questions and custom user queries.
- Provides a FastAPI-based REST API for interaction.
- Maintains conversation context using chat history.
- Logs operations for debugging and monitoring.

Prerequisites
-------------
- Python 3.8+
- Required Python packages (listed in requirements.txt):
  - langchain
  - langchain-community
  - langchain-google-genai
  - langgraph
  - fastapi
  - uvicorn
  - pandas
  - python-docx
  - python-dotenv
  - faiss-cpu
- A Google API key for Gemini models (set in a .env file).
- Input files:
  - .docx files in the `raw_files/docs` folder.
  - WhatsApp chat data in `raw_files/chat_with_intent.csv` with columns: Date, Time, Sender, Message, Keywords, Intent.

Installation
------------
1. Clone the repository:
   git clone <repository-url>
2. Navigate to the project directory:
   cd <project-directory>
3. Install dependencies:
   pip install -r requirements.txt
4. Set up environment variables:
   - Create a `.env` file in the project root.
   - Add your Google API key: `gemini_api_key=<your-api-key>`
5. Ensure the `raw_files/docs` folder contains .docx files and `raw_files/chat_with_intent.csv` exists with the required format.

Project Structure
-----------------
- `kb_docs.py`: Builds a knowledge base by processing .docx files, creating embeddings, and saving them to a FAISS vector store.
- `app_graph.py`: Implements the chatbot logic using LangGraph, handling document loading, WhatsApp chat parsing, and question answering.
- `api_chat.py`: Provides a FastAPI-based REST API for interacting with the chatbot.
- `raw_files/docs/`: Directory for .docx manual files.
- `raw_files/chat_with_intent.csv`: WhatsApp chat data in CSV format.
- `logs/`: Directory for log files (app.log, api.log).
- `hr_vector_store/`: Directory for the FAISS vector store.

Usage
-----
1. Build the Knowledge Base:
   - Run `kb_docs.py` to process .docx files and WhatsApp chat data, creating a FAISS vector store:
     python kb_docs.py
   - This generates the `hr_vector_store` directory.

2. Run the Chatbot CLI:
   - Execute `app_graph.py` to start the interactive CLI:
     python app_graph.py
   - Select a predefined question by number or enter a custom question. Type `quit` or `exit` to stop.

3. Run the API Server:
   - Start the FastAPI server with:
     python api_chat.py
   - The API runs on `http://127.0.0.1:8001`.
   - Endpoints:
     - GET `/predefined-questions`: Returns the list of predefined questions.
     - POST `/ask`: Submits a question (custom or predefined by index) and returns the answer.
       Example request body:
       {
         "question": "How to create a new user?",
         "predefined_index": null
       }
       or
       {
         "question": null,
         "predefined_index": 0
       }

Example API Usage
----------------
- Get predefined questions:
  curl http://127.0.0.1:8001/predefined-questions
- Ask a question:
  curl -X POST http://127.0.0.1:8001/ask -H "Content-Type: application/json" -d '{"question": "How to create a new user?"}'

Logging
-------
- Logs are saved to `logs/app.log` (for `kb_docs.py` and `app_graph.py`) and `logs/api.log` (for `api_chat.py`).
- Logs include timestamps, levels, and detailed messages for debugging.

Contributing
------------
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.

Contact
-------
For issues or questions, please open an issue on the GitHub repository or contact the maintainer.
