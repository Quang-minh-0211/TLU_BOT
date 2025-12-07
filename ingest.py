# src/ingest.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Đổi import này
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Cấu hình
DATA_PATH = "crawl_data"
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def load_documents():
    """Load tất cả file txt từ thư mục processed"""
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"✅ Đã load {len(documents)} documents")
    return documents

def split_documents(documents):
    """Chia documents thành chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Đã chia thành {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """Tạo ChromaDB từ chunks"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Đã lưu vào ChromaDB tại {CHROMA_PATH}")
    return vectorstore

def main():
    print("🚀 Bắt đầu ingest dữ liệu...")
    
    # 1. Load documents
    documents = load_documents()
    
    # 2. Split thành chunks
    chunks = split_documents(documents)
    
    # 3. Tạo vector store
    vectorstore = create_vector_store(chunks)
    
    print("🎉 Hoàn thành!")

if __name__ == "__main__":
    main()