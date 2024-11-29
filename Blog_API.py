import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

from fastapi.middleware.cors import CORSMiddleware

# Setup logging
logging.basicConfig(level=logging.INFO)

# FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Replace "*" with your frontend's URL in production, e.g., ["http://localhost:3000"]
    allow_origins=["http://localhost:3000", "https://www.learnaiwithus.codes"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Constants
DOC_PATH = "data"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "Blog_LLM"
PERSIST_DIRECTORY = "vector_db"

# Pydantic model for the input data


class QueryRequest(BaseModel):
    question: str


def ingest_pdf(doc_path):
    if os.path.exists(doc_path):
        file_paths = [os.path.join(doc_path, f)
                      for f in os.listdir(doc_path) if f.endswith('.pdf')]

        documents = []
        for file_path in file_paths:
            loader = UnstructuredPDFLoader(file_path)
            pdf_docs = loader.load()
            for doc in pdf_docs:
                doc.metadata = {"source": file_path}
                documents.append(doc)
        logging.info(f"Loaded {len(documents)} documents from {doc_path}")
        return documents
    else:
        logging.error(f"Path {doc_path} does not exist")
        return None


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def load_vector_db():
    ollama.pull(EMBEDDING_MODEL)
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None
        chunks = split_documents(data)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
    return vector_db


def create_retriever(vector_db, llm):
    QUERY_PROMPT = PromptTemplate(
        input_variables=['question'],
        template="""
    You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}
    """
    )
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    template = """
    Answer the question based ONLY on the following context. Do not use any external information.
    If the question contains only non-relevant words such as articles (e.g., 'is', 'and', 'the', 'what', etc.), provide no answer.
    If the question contains significant keywords, generate a brief explanation of the relevant concept and provide the formula if you find one else dont't.

    {context}

    Question:
    {question}

    just give definition and Explanation dont give unneccesarly response also provide Formula (if you find one else dont't.):
    
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logging.info("Chain created with preserved syntax.")
    return chain

# FastAPI endpoint to handle query requests


@app.post("/ask/")
async def ask_question(query_request: QueryRequest):
    llm = ChatOllama(model=MODEL_NAME)
    vector_db = load_vector_db()
    if vector_db is None:
        raise HTTPException(
            status_code=500, detail="Error loading vector database")

    retriever = create_retriever(vector_db, llm)
    chain = create_chain(retriever, llm)

    question = query_request.question
    try:
        # Get the response from the chain
        res = chain.invoke(input=question)
        return {"response": res}
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500, detail="Error processing the question")

# FastAPI app entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
