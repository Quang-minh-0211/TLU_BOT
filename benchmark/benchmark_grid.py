# benchmark/benchmark_grid_search.py
"""
Grid Search Benchmark - Test TẤT CẢ tổ hợp cấu hình
Tìm ra bộ cấu hình tối ưu nhất cho hệ thống RAG
"""

import json
import time
import os
import sys
import statistics
import warnings
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass, asdict
from itertools import product

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import Chroma
from langchain_community. document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError: 
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ============== CẤU HÌNH ==============
DATA_PATH = "data/processed"
TEST_DATASET_PATH = "/mnt/48AC6E9BAC6E82F4/Dev/TLUBot/evaluation/test_dataset.json"
RESULTS_PATH = "benchmark/results"

# ============== CÁC OPTIONS CẦN TEST ==============

EMBEDDING_OPTIONS = [
    {"name": "MiniLM", "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
    {"name": "E5-small", "model_id": "intfloat/multilingual-e5-small"},
    # {"name": "E5-base", "model_id": "intfloat/multilingual-e5-base"},  # Uncomment nếu có thời gian
]

CHUNKING_OPTIONS = [
    {"name": "small", "chunk_size": 300, "chunk_overlap": 50},
    {"name": "medium", "chunk_size": 500, "chunk_overlap": 100},
    {"name": "large", "chunk_size": 1000, "chunk_overlap": 200},
]

RETRIEVAL_OPTIONS = [
    {"name": "sim_k3", "search_type": "similarity", "k": 3},
    {"name": "sim_k5", "search_type": "similarity", "k": 5},
    {"name": "mmr_k5", "search_type": "mmr", "k": 5, "fetch_k": 10},
]

# ============== DATA CLASS ==============
@dataclass
class GridSearchResult:
    embedding_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_type: str
    retrieval_k: int
    hit_rate: float
    mrr: float
    total_time: float

# ============== HELPER FUNCTIONS ==============

def load_test_cases() -> List[Dict]:
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']

def load_documents():
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()

def check_match(doc_content: str, ground_truth:  str, threshold: float = 0.2) -> bool:
    """Kiểm tra document có match với ground truth không"""
    gt_words = set(ground_truth.lower().split())
    doc_words = set(doc_content.lower().split())
    
    if not gt_words: 
        return False
    
    overlap = len(gt_words & doc_words) / len(gt_words)
    return overlap > threshold

def evaluate_config(
    documents,
    test_cases,
    embedding_config: Dict,
    chunking_config: Dict,
    retrieval_config: Dict,
    num_tests: int = 10
) -> GridSearchResult:
    """Đánh giá một tổ hợp cấu hình"""
    
    start_time = time.time()
    
    try:
        # 1. Tạo embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_config['model_id'],
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config['chunk_size'],
            chunk_overlap=chunking_config['chunk_overlap']
        )
        chunks = text_splitter.split_documents(documents)
        
        # 3. Tạo vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
        
        # 4. Tạo retriever
        search_kwargs = {"k": retrieval_config['k']}
        if retrieval_config['search_type'] == 'mmr':
            search_kwargs['fetch_k'] = retrieval_config. get('fetch_k', 10)
        
        retriever = vectorstore.as_retriever(
            search_type=retrieval_config['search_type'],
            search_kwargs=search_kwargs
        )
        
        # 5. Evaluate
        hits = 0
        reciprocal_ranks = []
        
        for test_case in test_cases[: num_tests]:
            question = test_case['question']
            ground_truth = test_case['ground_truth']
            
            docs = retriever.invoke(question)
            
            found = False
            for rank, doc in enumerate(docs, 1):
                if check_match(doc.page_content, ground_truth):
                    hits += 1
                    reciprocal_ranks.append(1.0 / rank)
                    found = True
                    break
            
            if not found:
                reciprocal_ranks.append(0.0)
        
        hit_rate = hits / num_tests
        mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0
        
        # Cleanup
        del vectorstore
        del embeddings
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        hit_rate = 0
        mrr = 0
    
    total_time = time.time() - start_time
    
    return GridSearchResult(
        embedding_name=embedding_config['name'],
        embedding_model=embedding_config['model_id'],
        chunk_size=chunking_config['chunk_size'],
        chunk_overlap=chunking_config['chunk_overlap'],
        retrieval_type=retrieval_config['search_type'],
        retrieval_k=retrieval_config['k'],
        hit_rate=hit_rate,
        mrr=mrr,
        total_time=total_time
    )

# ============== MAIN GRID SEARCH ==============

def run_grid_search():
    """Chạy Grid Search trên tất cả tổ hợp"""
    
    print("\n" + "🔍" + "=" * 68)
    print("   GRID SEARCH - TÌM TỔ HỢP CẤU HÌNH TỐI ƯU")
    print("=" * 70)
    print(f"📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Tính số tổ hợp
    total_combinations = len(EMBEDDING_OPTIONS) * len(CHUNKING_OPTIONS) * len(RETRIEVAL_OPTIONS)
    print(f"\n📊 Số tổ hợp cần test: {total_combinations}")
    print(f"   - {len(EMBEDDING_OPTIONS)} Embedding models")
    print(f"   - {len(CHUNKING_OPTIONS)} Chunking configs")
    print(f"   - {len(RETRIEVAL_OPTIONS)} Retrieval configs")
    print("=" * 70)
    
    # Load data
    print("\n📚 Đang load dữ liệu...")
    documents = load_documents()
    test_cases = load_test_cases()
    print(f"   ✅ {len(documents)} documents, {len(test_cases)} test cases")
    
    # Grid Search
    results = []
    current = 0
    
    print(f"\n🚀 Bắt đầu Grid Search.. .\n")
    
    for emb_config in EMBEDDING_OPTIONS:
        for chunk_config in CHUNKING_OPTIONS:
            for ret_config in RETRIEVAL_OPTIONS:
                current += 1
                
                config_name = f"{emb_config['name']}_{chunk_config['name']}_{ret_config['name']}"
                print(f"   [{current}/{total_combinations}] Testing:  {config_name}")
                
                result = evaluate_config(
                    documents=documents,
                    test_cases=test_cases,
                    embedding_config=emb_config,
                    chunking_config=chunk_config,
                    retrieval_config=ret_config
                )
                
                results. append(result)
                print(f"      ✅ Hit Rate: {result.hit_rate:.2%}, MRR: {result.mrr:.3f}, Time: {result.total_time:.1f}s")
    
    return results

def print_grid_results(results: List[GridSearchResult]):
    """In kết quả Grid Search"""
    
    print(f"\n{'='*90}")
    print("📊 KẾT QUẢ GRID SEARCH - TẤT CẢ TỔ HỢP")
    print(f"{'='*90}")
    
    # Sort by MRR
    sorted_results = sorted(results, key=lambda x: x.mrr, reverse=True)
    
    print(f"\n{'Rank':<6} {'Embedding':<12} {'Chunk':<12} {'Retrieval':<12} {'Hit Rate':<12} {'MRR':<10} {'Time':<8}")
    print("-" * 90)
    
    for i, r in enumerate(sorted_results):
        rank = i + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank: >2}"
        
        chunk_str = f"{r.chunk_size}"
        ret_str = f"{r.retrieval_type}_k{r.retrieval_k}"
        hit_str = f"{r.hit_rate:.2%}"
        mrr_str = f"{r.mrr:.3f}"
        time_str = f"{r.total_time:.1f}s"
        
        print(f"{medal:<6} {r.embedding_name:<12} {chunk_str:<12} {ret_str:<12} {hit_str:<12} {mrr_str:<10} {time_str:<8}")
    
    # Best config
    best = sorted_results[0]
    print(f"\n{'='*90}")
    print("🏆 CẤU HÌNH TỐT NHẤT")
    print(f"{'='*90}")
    print(f"""
   📌 Embedding:   {best.embedding_name}
      Model:  {best.embedding_model}
   
   📌 Chunking: 
      Chunk Size: {best.chunk_size}
      Chunk Overlap: {best.chunk_overlap}
   
   📌 Retrieval:
      Type: {best.retrieval_type}
      K: {best. retrieval_k}
   
   📊 Kết quả:
      Hit Rate: {best.hit_rate:.2%}
      MRR: {best.mrr:.3f}
""")
    print(f"{'='*90}")
    
    return best

def print_analysis(results: List[GridSearchResult]):
    """Phân tích kết quả theo từng component"""
    
    print(f"\n{'='*90}")
    print("📈 PHÂN TÍCH THEO TỪNG COMPONENT")
    print(f"{'='*90}")
    
    # Phân tích Embedding
    print("\n1️⃣  EMBEDDING MODELS (Trung bình qua các config khác)")
    print("-" * 50)
    
    emb_scores = {}
    for r in results:
        if r.embedding_name not in emb_scores:
            emb_scores[r.embedding_name] = []
        emb_scores[r.embedding_name].append(r.mrr)
    
    for name, scores in sorted(emb_scores.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        avg_mrr = statistics.mean(scores)
        print(f"   {name: <20} Avg MRR: {avg_mrr:.3f}")
    
    # Phân tích Chunking
    print("\n2️⃣  CHUNKING STRATEGIES (Trung bình qua các config khác)")
    print("-" * 50)
    
    chunk_scores = {}
    for r in results:
        key = f"size={r.chunk_size}"
        if key not in chunk_scores:
            chunk_scores[key] = []
        chunk_scores[key].append(r. mrr)
    
    for name, scores in sorted(chunk_scores.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        avg_mrr = statistics.mean(scores)
        print(f"   {name:<20} Avg MRR: {avg_mrr:.3f}")
    
    # Phân tích Retrieval
    print("\n3️⃣  RETRIEVAL STRATEGIES (Trung bình qua các config khác)")
    print("-" * 50)
    
    ret_scores = {}
    for r in results: 
        key = f"{r.retrieval_type}_k{r.retrieval_k}"
        if key not in ret_scores:
            ret_scores[key] = []
        ret_scores[key].append(r. mrr)
    
    for name, scores in sorted(ret_scores.items(), key=lambda x: statistics.mean(x[1]), reverse=True):
        avg_mrr = statistics.mean(scores)
        print(f"   {name:<20} Avg MRR: {avg_mrr:.3f}")
    
    print(f"\n{'='*90}")

def save_grid_results(results: List[GridSearchResult], best:  GridSearchResult):
    """Lưu kết quả"""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_PATH, f"grid_search_{timestamp}.json")
    
    data = {
        'timestamp': timestamp,
        'total_combinations': len(results),
        'best_config':  asdict(best),
        'all_results': [asdict(r) for r in results]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu:  {filename}")

# ============== MAIN ==============

def main():
    # Chạy Grid Search
    results = run_grid_search()
    
    # In kết quả
    best = print_grid_results(results)
    
    # Phân tích
    print_analysis(results)
    
    # Lưu
    save_grid_results(results, best)
    
    print("\n" + "🎉" + "=" * 68)
    print("   GRID SEARCH HOÀN TẤT!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()