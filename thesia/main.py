from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma  # Ensure this import is updated
from langchain.embeddings import HuggingFaceEmbeddings  # Ensure this import is updated
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from typing import Dict
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag.llm_config import create_llm

# Load environment variables from .env file
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        self.data_dir = Path("data")
        self.pdf_dir = self.data_dir / "pdfs"
        self.notes_dir = self.data_dir / "notes"
        self.vectorstore_dir = self.data_dir / "vectorstore"
        
        # Create directories if they don't exist
        for dir in [self.data_dir, self.pdf_dir, self.notes_dir, self.vectorstore_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Ensure this is updated
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def load_documents(self):
        documents = []
        
        # Load PDFs
        for pdf in self.pdf_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf))
            documents.extend(loader.load())
            
        # Load text files
        for txt in self.notes_dir.glob("*.txt"):
            loader = TextLoader(str(txt))
            documents.extend(loader.load())
            
        # Load Word documents
        for doc in self.notes_dir.glob("*.doc*"):
            loader = UnstructuredWordDocumentLoader(str(doc))
            documents.extend(loader.load())
            
        return documents

class RAGSystem:
    def __init__(self):
        self.processor = DocumentProcessor()
        # Retrieve the API key from the environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        # Pass the API key to the create_llm function
        self.llm = create_llm(self.api_key)  # Ensure this is updated
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,  # This ensures the memory returns the conversation history
            output_key="answer"  # Explicitly set the output key for memory
        )
        self.vectorstore = None
        self.qa_chain = None
        
    def initialize(self):
        # Load and process documents
        docs = self.processor.load_documents()
        chunks = self.processor.text_splitter.split_documents(docs)
        
        # Create or load vectorstore
        self.vectorstore = Chroma(
            persist_directory=str(self.processor.vectorstore_dir),
            embedding_function=self.processor.embeddings
        )
        
        # Add documents if vectorstore is empty
        if len(self.vectorstore.get()) == 0:
            self.vectorstore.add_documents(chunks)
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"  # Explicitly set the output key
        )
        
    def query(self, question: str) -> Dict:
        if not self.qa_chain:
            raise ValueError("System not initialized. Call initialize() first.")
            
        result = self.qa_chain.invoke({"question": question})  # Ensure this is updated
        
        # Format sources
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", None)
            })
            
        return {
            "answer": result["answer"],
            "sources": sources
        }

if __name__ == "__main__":
    rag = RAGSystem()
    rag.initialize()
    
    # Example query
    result = rag.query("What are the key considerations for general anesthesia in elderly patients?")
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source']} (Page {source['page']})")