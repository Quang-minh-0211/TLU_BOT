# chat. py - Phiên bản mới với LCEL
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Prompt template cho chatbot tư vấn tuyển sinh
PROMPT_TEMPLATE = """
Bạn là TLUBot - trợ lý tư vấn tuyển sinh của Trường Đại học Thủy lợi. 
Hãy trả lời câu hỏi dựa trên thông tin được cung cấp bên dưới.
Nếu không tìm thấy thông tin, hãy nói rằng bạn không có thông tin về vấn đề này. 
Trả lời bằng tiếng Việt, thân thiện và chính xác.

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Trả lời:
"""

def load_vectorstore():
    """Load ChromaDB đã tạo"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectorstore

def format_docs(docs):
    """Format documents thành string"""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore):
    """Tạo RAG chain với LCEL"""
    # Khởi tạo LLM
    llm = OllamaLLM(model="qwen2.5:7b")
    
    # Tạo retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Tạo prompt
    prompt = ChatPromptTemplate. from_template(PROMPT_TEMPLATE)
    
    # Tạo RAG chain với LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def chat(rag_chain, question):
    """Trả lời câu hỏi"""
    response = rag_chain.invoke(question)
    return response

def main():
    print("🤖 TLU Chatbot - Tư vấn tuyển sinh Đại học Thủy lợi")
    print("=" * 50)
    print("💡 Gõ 'exit' hoặc 'q' để thoát\n")
    
    # Load vector store và tạo chain
    print("⏳ Đang khởi tạo chatbot...")
    vectorstore = load_vectorstore()
    rag_chain, retriever = create_rag_chain(vectorstore)
    print("✅ Sẵn sàng!\n")
    
    while True:
        question = input("👤 Bạn: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Tạm biệt!")
            break
        
        if not question:
            continue
        
        print("\n⏳ Đang xử lý...")
        answer = chat(rag_chain, question)
        print(f"\n🤖 TLUBot: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()