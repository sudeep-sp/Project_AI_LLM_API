import os
import logging
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


logging.basicConfig(level=logging.INFO)

DOC_PATH = "data"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "Blog_LLM"
PERSIST_DIRECTORY = "vector_db"


def ingest_pdf(doc_path):

    if os.path.exists(doc_path):
        file_paths = [os.path.join(doc_path, f)
                      for f in os.listdir(doc_path) if f.endswith('.pdf')]

        documents = []
        for file_path in file_paths:
            # Create loader for each PDF
            loader = UnstructuredPDFLoader(file_path)
            pdf_docs = loader.load()  # Load the documents
            for doc in pdf_docs:
                # Add metadata with the file path as the source
                doc.metadata = {"source": file_path}
                documents.append(doc)  # Append to the final documents list
        logging.info(f"Loaded {len(documents)} documents from {doc_path}")
        return documents

    else:
        logging.error(f"Path {doc_path} does not exist")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def load_vector_db():
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    # ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=PERSIST_DIRECTORY,
        )
        logging.info("Loaded existing vector database.")
    else:
        # Load and process the PDF document
        data = ingest_pdf(DOC_PATH)
        if data is None:
            return None

        # Split the documents into chunks
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
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=['question'],
        template="""
    You are an AI language model assistant. Your task is to generate five diffrent versions of the given user question to retrive relevant documents from
    a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitation of the distance-based
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
    # RAG prompt
    template = """
    Answer the question based ONLY on the following context. Do not use any external information.
    If the question contains only non-relevant words such as articles (e.g., 'is', 'and', 'the', 'what', etc.), provide no answer.
    If the question contains significant keywords, generate a brief explanation of the relevant concept and provide the formula if you find one else dont't.

    {context}

    Question:
    {question}

    Explanation (brief) and Formula (if you find one else dont't.):
    
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


def main():

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the vector database
    vector_db = load_vector_db()
    if vector_db is None:
        return

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = input("enter your keyword to Gen: ")

    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()
