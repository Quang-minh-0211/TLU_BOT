# test_retriever.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_DIR = "db/tlu_chroma"

def test_retrieval(query: str, k: int = 5):
    """Test xem retriever tìm được gì"""
    
    print(f"\n{'='*80}")
    print(f"🔍 TEST QUERY: '{query}'")
    print(f"{'='*80}\n")
    
    # Load embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Load vector DB
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="tlu_data"
    )
    
    # 🔧 METHOD 1: Similarity search (có score)
    print("📊 Method 1: Similarity Search with Scores")
    print("-"*80)
    docs_with_scores = vectordb.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n📄 Result {i} (Score: {score:.4f}):")
        print(f"   Source: {doc.metadata.get('source', 'N/A')}")
        print(f"   Content: {doc.page_content[:1000]}...")
        print(f"   Full Metadata: {doc.metadata}")
    
    print("\n" + "="*80)
    
    # 🔧 METHOD 2: MMR (Maximum Marginal Relevance) - giảm trùng lặp
    print("\n📊 Method 2: MMR Search (Đa dạng hóa kết quả)")
    print("-"*80)
    docs_mmr = vectordb.max_marginal_relevance_search(query, k=k, fetch_k=20)
    
    for i, doc in enumerate(docs_mmr, 1):
        print(f"\n📄 Result {i}:")
        print(f"   Source: {doc.metadata.get('source', 'N/A')}")
        print(f"   Content: {doc.page_content[:1000]}...")
    
    print("\n" + "="*80 + "\n")


def test_multiple_queries():
    """Test nhiều queries để so sánh"""
    
    test_cases = [
        "Hiệu trưởng hiện tại trường Đại Học Thủy lợi"
        
    ]
    
    for query in test_cases:
        test_retrieval(query, k=3)
        input("\n⏸️  Nhấn Enter để tiếp tục test query tiếp theo...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Chạy với query cụ thể: python test_retriever.py "học phí"
        query = " ".join(sys.argv[1:])
        test_retrieval(query, k=5)
    else:
        # Chạy test suite
        test_multiple_queries()