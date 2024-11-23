from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentProcessor:
    def __init__(self):
        self.data_dir = Path("data")
        self.pdf_dir = self.data_dir / "pdfs"
        self.notes_dir = self.data_dir / "notes"
        self.vectorstore_dir = self.data_dir / "vectorstore"
        
        # Create directories if they don't exist
        for dir in [self.data_dir, self.pdf_dir, self.notes_dir, self.vectorstore_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

if __name__ == "__main__":
    processor = DocumentProcessor()
    docs = processor.load_documents()
    print(f"Loaded {len(docs)} documents")