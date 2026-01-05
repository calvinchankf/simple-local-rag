"""
RAG (Retrieval-Augmented Generation) System using Qwen 3 and Ollama

This script demonstrates how to:
1. Load PDF documents
2. Split them into chunks
3. Create embeddings
4. Store in a vector database (ChromaDB)
5. Query the documents with natural language
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
DATA_PATH = "data/"
PDF_FILENAME = "llama2.pdf"  # Change this to your PDF filename
CHROMA_PATH = "chroma_db"


def load_documents():
    """Load PDF document from the data directory."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. "
            f"Please place your PDF in the '{DATA_PATH}' directory "
            f"or update PDF_FILENAME in the script."
        )

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} page(s) from {PDF_FILENAME}")
    return documents


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"✓ Split into {len(all_splits)} chunks")
    return all_splits


def get_embedding_function(model_name="all-MiniLM-L6-v2"):
    """
    Create embedding function using HuggingFace Sentence Transformers.

    The model will be downloaded automatically on first run.
    Popular models:
    - "all-MiniLM-L6-v2" (default): Fast and efficient, good for most use cases
    - "all-mpnet-base-v2": Higher quality, slightly slower
    - "paraphrase-multilingual-MiniLM-L12-v2": For multilingual support

    Note: First run will download the model (~90MB for all-MiniLM-L6-v2)
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def index_documents(chunks, embedding_function):
    """Create vector store and index document chunks."""
    print(f"⏳ Indexing {len(chunks)} chunks... (this may take a moment)")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    vectorstore.persist()

    print(f"✓ Vector database saved to: {CHROMA_PATH}")
    return vectorstore


def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    """
    Create the RAG chain that connects retrieval and generation.

    Args:
        vector_store: ChromaDB vector store
        llm_model_name: Qwen model to use (qwen3:8b recommended)
        context_window: Context window size for the model
    """
    # Initialize the LLM
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,  # Set to 0 for consistent answers
        num_ctx=context_window
    )

    # Create retriever (fetches relevant chunks)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}  # Retrieve top 3 most relevant chunks
    )

    # Define the prompt template
    template = """Answer the question based ONLY on the following context:

{context}

Question: {question}

Answer: """

    prompt = ChatPromptTemplate.from_template(template)

    # Build the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def query_rag(chain, question):
    """Query the RAG system with a question."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    response = chain.invoke(question)

    print(f"Response:\n{response}")
    print(f"{'='*60}\n")
    return response


def main():
    """Main execution function."""
    print("\n🚀 Starting RAG System...\n")

    # Step 1-2: Load and split documents
    docs = load_documents()
    chunks = split_documents(docs)

    # Step 3-4: Create embeddings
    print("⏳ Loading embedding model...")
    embedding_function = get_embedding_function()
    print("✓ Embedding model loaded")

    # Step 5-6: Index documents in vector store
    vector_store = index_documents(chunks, embedding_function)

    # Step 7: Create RAG chain
    print("⏳ Building RAG chain...")
    rag_chain = create_rag_chain(vector_store)
    print("✓ RAG chain ready\n")

    # Step 8: Query the system
    print("📝 Running sample query...\n")

    # Run one sample query to demonstrate
    query_rag(rag_chain, "What is the main topic of this document?")

    # Interactive query loop
    print("\n✅ RAG system ready!")
    print("💡 You can now ask questions about your document")
    print("💡 Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            # Get user input
            user_question = input("Your question: ").strip()

            # Check for exit commands
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Skip empty questions
            if not user_question:
                continue

            # Query the RAG system
            query_rag(rag_chain, user_question)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue


if __name__ == "__main__":
    main()
