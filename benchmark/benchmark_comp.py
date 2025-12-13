# benchmark/benchmark_components.py
"""
Benchmark toàn diện tất cả thành phần RAG
- Embedding Models
- Chunking Strategies  
- Retrieval Strategies

Phiên bản: 2.0 - Đã tối ưu, không lỗi
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

# Tắt warnings
warnings.filterwarnings("ignore")

# Thêm path
sys.path.append(os.path.dirname(os. path.dirname(os.path. abspath(__file__))))

# ============== IMPORTS ==============
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import HuggingFaceEmbeddings - thử bản mới trước
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# ============== CẤU HÌNH ==============
DATA_PATH = "data/processed"
TEST_DATASET_PATH = "/mnt/48AC6E9BAC6E82F4/Dev/TLUBot/evaluation/test_dataset.json"
RESULTS_PATH = "benchmark/results"

# Thư mục chứa tất cả ChromaDB tạm - sẽ không xóa trong quá trình chạy
BENCHMARK_TEMP_DIR = "benchmark/temp_dbs"

# ============== CẤU HÌNH TEST ==============

EMBEDDING_MODELS = [
    {
        "name": "multilingual-MiniLM",
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    {
        "name": "multilingual-e5-small",
        "model_id":  "intfloat/multilingual-e5-small",
    },
    # Uncomment nếu muốn test thêm (tốn thời gian)
    # {
    #     "name":  "multilingual-e5-base",
    #     "model_id": "intfloat/multilingual-e5-base",
    # },
]

CHUNKING_CONFIGS = [
    {"name": "small", "chunk_size": 300, "chunk_overlap": 50},
    {"name": "medium", "chunk_size": 500, "chunk_overlap": 100},
    {"name": "large", "chunk_size": 1000, "chunk_overlap": 200},
    {"name": "xlarge", "chunk_size": 1500, "chunk_overlap": 300},
]

RETRIEVAL_CONFIGS = [
    {"name": "similarity_k3", "search_type": "similarity", "k": 3},
    {"name": "similarity_k5", "search_type": "similarity", "k": 5},
    {"name": "similarity_k7", "search_type": "similarity", "k": 7},
    {"name": "mmr_k5", "search_type": "mmr", "k": 5, "fetch_k": 10},
    {"name": "mmr_k7", "search_type": "mmr", "k": 7, "fetch_k": 15},
]

# ============== DATA CLASS ==============
@dataclass
class BenchmarkResult:
    component_type: str
    config_name: str
    config_details: Dict
    hit_rate:  float
    mrr: float
    avg_latency: float
    num_chunks: int
    index_time: float

# ============== HELPER FUNCTIONS ==============

def load_test_cases() -> List[Dict]:
    """Load test dataset"""
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']

def load_documents():
    """Load documents"""
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    return loader.load()

def calculate_hit_and_mrr(retriever, test_cases: List[Dict], num_tests: int = 10):
    """Tính Hit Rate và MRR"""
    hit_count = 0
    reciprocal_ranks = []
    latencies = []
    
    for test_case in test_cases[:num_tests]:
        question = test_case['question']
        ground_truth = test_case['ground_truth'].lower()
        gt_words = set(ground_truth.split())
        
        # Đo latency
        start = time.time()
        docs = retriever.invoke(question)
        latencies.append(time.time() - start)
        
        # Tìm hit
        found = False
        for rank, doc in enumerate(docs, 1):
            doc_words = set(doc.page_content.lower().split())
            overlap = len(gt_words & doc_words) / len(gt_words) if gt_words else 0
            
            if overlap > 0.2 and not found:
                hit_count += 1
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break
        
        if not found:
            reciprocal_ranks.append(0.0)
    
    hit_rate = hit_count / num_tests
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0
    avg_latency = statistics.mean(latencies) if latencies else 0
    
    return hit_rate, mrr, avg_latency

# ============== BENCHMARK FUNCTIONS ==============

def benchmark_embeddings(documents, test_cases) -> List[BenchmarkResult]: 
    """Benchmark các Embedding Models"""
    print("\n" + "=" * 60)
    print("1️⃣  BENCHMARK EMBEDDING MODELS")
    print("=" * 60)
    
    results = []
    
    # Chunking cố định
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    for i, config in enumerate(EMBEDDING_MODELS):
        print(f"\n   🔤 [{i+1}/{len(EMBEDDING_MODELS)}] Testing: {config['name']}")
        
        try:
            # Tạo embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=config['model_id'],
                model_kwargs={'device': 'cpu'}
            )
            
            # Tạo vectorstore trong memory (không persist)
            start_index = time.time()
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                # Không dùng persist_directory để tránh lỗi lock
            )
            index_time = time.time() - start_index
            
            # Benchmark
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            hit_rate, mrr, avg_latency = calculate_hit_and_mrr(retriever, test_cases)
            
            result = BenchmarkResult(
                component_type="embedding",
                config_name=config['name'],
                config_details=config,
                hit_rate=hit_rate,
                mrr=mrr,
                avg_latency=avg_latency,
                num_chunks=len(chunks),
                index_time=index_time
            )
            results.append(result)
            
            print(f"      ✅ Hit Rate: {hit_rate:.2%}, MRR: {mrr:.3f}, Index:  {index_time:.2f}s")
            
            # Cleanup
            del vectorstore
            del embeddings
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append(BenchmarkResult(
                component_type="embedding",
                config_name=config['name'],
                config_details=config,
                hit_rate=0, mrr=0, avg_latency=0, num_chunks=0, index_time=0
            ))
    
    return results

def benchmark_chunking(documents, test_cases) -> List[BenchmarkResult]:
    """Benchmark các Chunking Strategies"""
    print("\n" + "=" * 60)
    print("2️⃣  BENCHMARK CHUNKING STRATEGIES")
    print("=" * 60)
    
    results = []
    
    # Embedding cố định
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    for i, config in enumerate(CHUNKING_CONFIGS):
        print(f"\n   📄 [{i+1}/{len(CHUNKING_CONFIGS)}] Testing: {config['name']} (size={config['chunk_size']})")
        
        try:
            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['chunk_overlap']
            )
            chunks = text_splitter.split_documents(documents)
            
            # Tạo vectorstore
            start_index = time.time()
            vectorstore = Chroma. from_documents(
                documents=chunks,
                embedding=embeddings,
            )
            index_time = time.time() - start_index
            
            # Benchmark
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            hit_rate, mrr, avg_latency = calculate_hit_and_mrr(retriever, test_cases)
            
            result = BenchmarkResult(
                component_type="chunking",
                config_name=config['name'],
                config_details=config,
                hit_rate=hit_rate,
                mrr=mrr,
                avg_latency=avg_latency,
                num_chunks=len(chunks),
                index_time=index_time
            )
            results.append(result)
            
            print(f"      ✅ Chunks: {len(chunks)}, Hit Rate: {hit_rate:.2%}, MRR: {mrr:.3f}")
            
            # Cleanup
            del vectorstore
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append(BenchmarkResult(
                component_type="chunking",
                config_name=config['name'],
                config_details=config,
                hit_rate=0, mrr=0, avg_latency=0, num_chunks=0, index_time=0
            ))
    
    del embeddings
    return results

def benchmark_retrieval(documents, test_cases) -> List[BenchmarkResult]:
    """Benchmark các Retrieval Strategies"""
    print("\n" + "=" * 60)
    print("3️⃣  BENCHMARK RETRIEVAL STRATEGIES")
    print("=" * 60)
    
    results = []
    
    # Setup cố định
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    print(f"\n   📚 Tạo vectorstore với {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    for i, config in enumerate(RETRIEVAL_CONFIGS):
        print(f"\n   🔍 [{i+1}/{len(RETRIEVAL_CONFIGS)}] Testing: {config['name']}")
        
        try: 
            # Tạo retriever
            search_kwargs = {"k": config['k']}
            if config['search_type'] == 'mmr':
                search_kwargs['fetch_k'] = config. get('fetch_k', 10)
            
            retriever = vectorstore.as_retriever(
                search_type=config['search_type'],
                search_kwargs=search_kwargs
            )
            
            # Benchmark
            hit_rate, mrr, avg_latency = calculate_hit_and_mrr(retriever, test_cases)
            
            result = BenchmarkResult(
                component_type="retrieval",
                config_name=config['name'],
                config_details=config,
                hit_rate=hit_rate,
                mrr=mrr,
                avg_latency=avg_latency,
                num_chunks=len(chunks),
                index_time=0
            )
            results.append(result)
            
            print(f"      ✅ Hit Rate:  {hit_rate:.2%}, MRR: {mrr:. 3f}, Latency: {avg_latency:.3f}s")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append(BenchmarkResult(
                component_type="retrieval",
                config_name=config['name'],
                config_details=config,
                hit_rate=0, mrr=0, avg_latency=0, num_chunks=0, index_time=0
            ))
    
    del vectorstore
    del embeddings
    return results

# ============== PRINT FUNCTIONS ==============

def print_results(results: List[BenchmarkResult]):
    """In kết quả đẹp"""
    
    # Embedding results
    emb_results = [r for r in results if r.component_type == "embedding"]
    if emb_results:
        print(f"\n{'='*70}")
        print("📊 KẾT QUẢ:  EMBEDDING MODELS")
        print(f"{'='*70}")
        print(f"\n{'Config':<25} {'Hit Rate':<12} {'MRR':<10} {'Index Time':<12}")
        print("-" * 60)
        
        sorted_emb = sorted(emb_results, key=lambda x:  x.mrr, reverse=True)
        for i, r in enumerate(sorted_emb):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{medal} {r.config_name:<22} {r.hit_rate:<12.2%} {r.mrr:<10.3f} {r.index_time:<12.2f}s")
    
    # Chunking results
    chunk_results = [r for r in results if r.component_type == "chunking"]
    if chunk_results:
        print(f"\n{'='*70}")
        print("📊 KẾT QUẢ:  CHUNKING STRATEGIES")
        print(f"{'='*70}")
        print(f"\n{'Config':<12} {'Size':<10} {'Overlap':<10} {'Chunks':<10} {'Hit Rate':<12} {'MRR':<10}")
        print("-" * 70)
        
        sorted_chunk = sorted(chunk_results, key=lambda x: x.mrr, reverse=True)
        for i, r in enumerate(sorted_chunk):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            size = r.config_details.get('chunk_size', 0)
            overlap = r.config_details.get('chunk_overlap', 0)
            print(f"{medal} {r.config_name:<9} {size:<10} {overlap:<10} {r.num_chunks:<10} {r. hit_rate:<12.2%} {r.mrr:<10.3f}")
    
    # Retrieval results
    ret_results = [r for r in results if r.component_type == "retrieval"]
    if ret_results:
        print(f"\n{'='*70}")
        print("📊 KẾT QUẢ:  RETRIEVAL STRATEGIES")
        print(f"{'='*70}")
        print(f"\n{'Config':<18} {'Type':<12} {'K':<6} {'Hit Rate':<12} {'MRR':<10} {'Latency':<10}")
        print("-" * 70)
        
        sorted_ret = sorted(ret_results, key=lambda x: x.mrr, reverse=True)
        for i, r in enumerate(sorted_ret):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            stype = r.config_details. get('search_type', '')
            k = r.config_details.get('k', 0)
            print(f"{medal} {r.config_name:<15} {stype:<12} {k:<6} {r.hit_rate:<12.2%} {r.mrr:<10.3f} {r.avg_latency:<10.3f}s")

def print_recommendations(results: List[BenchmarkResult]):
    """In khuyến nghị"""
    print(f"\n{'='*70}")
    print("🎯 KHUYẾN NGHỊ CẤU HÌNH TỐI ƯU")
    print(f"{'='*70}")
    
    # Best embedding
    emb = [r for r in results if r. component_type == "embedding" and r.mrr > 0]
    if emb:
        best = max(emb, key=lambda x: x.mrr)
        print(f"\n1️⃣  EMBEDDING MODEL:")
        print(f"   ✅ {best.config_name}")
        print(f"      Model: {best.config_details. get('model_id', 'N/A')}")
        print(f"      MRR: {best.mrr:.3f} | Hit Rate: {best.hit_rate:.2%}")
    
    # Best chunking
    chunk = [r for r in results if r.component_type == "chunking" and r.mrr > 0]
    if chunk: 
        best = max(chunk, key=lambda x: x.mrr)
        print(f"\n2️⃣  CHUNKING STRATEGY:")
        print(f"   ✅ {best.config_name}")
        print(f"      Chunk Size: {best.config_details['chunk_size']} | Overlap: {best. config_details['chunk_overlap']}")
        print(f"      MRR: {best.mrr:.3f} | Hit Rate: {best.hit_rate:.2%}")
    
    # Best retrieval
    ret = [r for r in results if r. component_type == "retrieval" and r.mrr > 0]
    if ret:
        best = max(ret, key=lambda x: x.mrr)
        print(f"\n3️⃣  RETRIEVAL STRATEGY:")
        print(f"   ✅ {best.config_name}")
        print(f"      Type: {best.config_details['search_type']} | K: {best.config_details['k']}")
        print(f"      MRR: {best.mrr:.3f} | Hit Rate: {best.hit_rate:.2%}")
    
    print(f"\n{'='*70}")

def save_results(results: List[BenchmarkResult]):
    """Lưu kết quả"""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_PATH, f"components_benchmark_{timestamp}.json")
    
    data = {
        'timestamp': timestamp,
        'num_configs': len(results),
        'results': [asdict(r) for r in results]
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu:  {filename}")

# ============== MAIN ==============

def main():
    print("\n" + "🚀" + "=" * 68)
    print("   BENCHMARK TOÀN DIỆN CÁC THÀNH PHẦN RAG")
    print("=" * 70)
    print(f"📅 Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Data path: {DATA_PATH}")
    print(f"📋 Test file: {TEST_DATASET_PATH}")
    print("=" * 70)
    
    # Load data
    print("\n📚 Đang load dữ liệu...")
    documents = load_documents()
    test_cases = load_test_cases()
    print(f"   ✅ {len(documents)} documents, {len(test_cases)} test cases")
    
    all_results = []
    
    # 1. Benchmark Embeddings
    emb_results = benchmark_embeddings(documents, test_cases)
    all_results.extend(emb_results)
    
    # 2. Benchmark Chunking
    chunk_results = benchmark_chunking(documents, test_cases)
    all_results.extend(chunk_results)
    
    # 3. Benchmark Retrieval
    ret_results = benchmark_retrieval(documents, test_cases)
    all_results.extend(ret_results)
    
    # Print results
    print_results(all_results)
    print_recommendations(all_results)
    
    # Save
    save_results(all_results)
    
    print("\n" + "🎉" + "=" * 68)
    print("   BENCHMARK HOÀN TẤT!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()