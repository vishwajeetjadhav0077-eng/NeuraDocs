import os
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag import (
    load_pdf,
    split_documents,
    create_embeddings,
    create_vector_store,
    create_retriever,
    create_llm,
    create_prompt,
    answer_question
)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="NeuraDocs API",
    description="AI-powered multi-document RAG API",
    version="1.0.0"
)


# --------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------

retriever = None
llm = None
prompt = None

uploaded_documents = []


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class QuestionRequest(BaseModel):

    question: str


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "NeuraDocs API is running",
        "status": "active"
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "documents_loaded": len(uploaded_documents)
    }


# --------------------------------------------------
# UPLOAD DOCUMENTS
# --------------------------------------------------

@app.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...)
):

    global retriever
    global llm
    global prompt
    global uploaded_documents


    if not files:

        raise HTTPException(
            status_code=400,
            detail="No PDF files uploaded."
        )


    all_documents = []


    os.makedirs(
        "data/uploads",
        exist_ok=True
    )


    for file in files:

        # Check file type

        if not file.filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=f"{file.filename} is not a PDF."
            )


        file_path = os.path.join(
            "data/uploads",
            file.filename
        )


        # Save uploaded file

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        # Load PDF

        documents = load_pdf(
            file_path
        )


        # Store filename in metadata

        for document in documents:

            document.metadata["source"] = (
                file.filename
            )


        all_documents.extend(
            documents
        )


    # --------------------------------------------------
    # CHUNK DOCUMENTS
    # --------------------------------------------------

    chunks = split_documents(
        all_documents
    )


    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    embeddings = create_embeddings()


    # --------------------------------------------------
    # CREATE VECTOR STORE
    # --------------------------------------------------

    vector_store = create_vector_store(
        chunks,
        embeddings
    )


    # --------------------------------------------------
    # CREATE RETRIEVER
    # --------------------------------------------------

    retriever = create_retriever(
        vector_store
    )


    # --------------------------------------------------
    # CREATE LLM
    # --------------------------------------------------

    llm = create_llm()


    # --------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------

    prompt = create_prompt()


    # Store filenames

    uploaded_documents = [
        file.filename
        for file in files
    ]


    return {

        "message": "Documents uploaded and indexed successfully",

        "documents": uploaded_documents,

        "chunks": len(chunks),

        "status": "ready"
    }


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    if retriever is None:

        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet."
        )


    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    answer, sources = answer_question(
        request.question,
        retriever,
        llm,
        prompt
    )


    return {

        "question": request.question,

        "answer": answer,

        "sources": sources
    }