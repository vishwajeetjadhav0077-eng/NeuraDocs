from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate


def load_pdf(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"Loaded {len(documents)} pages")

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


def create_vector_store(chunks, embeddings):

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


def create_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )

    return retriever


def create_llm():

    llm = Ollama(
        model="llama3.2:3b",
        base_url="http://host.docker.internal:11434"
    )

    return llm

def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
You are NeuraDocs, an intelligent document analysis assistant.

You answer questions using the retrieved information from
one or more uploaded documents.

IMPORTANT RULES:

1. Use ONLY the information provided in the retrieved context.
2. Never invent information.
3. Pay close attention to the DOCUMENT name.
4. When comparing documents, clearly separate the information
   belonging to each document.
5. If the user asks about "both", "these documents", or
   "the two resumes", compare the uploaded documents.
6. If information exists in only one document, clearly say so.
7. If the retrieved context does not contain enough information,
   say that you could not find enough information.
8. For comparison questions, use a clear table or bullet points
   when appropriate.

Retrieved Context:

{context}

User Question:

{question}

Answer:
"""
    )

    return prompt


def answer_question(
    question,
    retriever,
    llm,
    prompt
):

    documents = retriever.invoke(question)

    print("\n========== RETRIEVED DOCUMENTS ==========")

    for i, document in enumerate(documents):

        filename = document.metadata.get(
            "source",
            "Unknown"
        )

        page = (
            document.metadata.get(
                "page",
                0
            ) + 1
        )

        print(f"\n--- Chunk {i + 1} ---")
        print(f"File: {filename}")
        print(f"Page: {page}")
        print(document.page_content[:1000])

    print("\n==========================================\n")


    # --------------------------------------------------
    # BUILD SOURCE-AWARE CONTEXT
    # --------------------------------------------------

    context_parts = []

    for document in documents:

        filename = document.metadata.get(
            "source",
            "Unknown"
        )

        page = (
            document.metadata.get(
                "page",
                0
            ) + 1
        )

        context_parts.append(
            f"""
[DOCUMENT: {filename}]
[PAGE: {page}]

{document.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )


    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    response = llm.invoke(
        formatted_prompt
    )


    # --------------------------------------------------
    # CREATE SOURCES
    # --------------------------------------------------

    sources = []

    for document in documents:

        filename = document.metadata.get(
            "source",
            "Unknown"
        )

        page_number = (
            document.metadata.get(
                "page",
                0
            ) + 1
        )

        sources.append(
            {
                "file": filename,
                "page": page_number,
                "content": document.page_content
            }
        )


    return response, sources