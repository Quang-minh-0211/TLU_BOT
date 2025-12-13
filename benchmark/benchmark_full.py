# benchmark/benchmark_full.py
"""
Benchmark toàn diện cho TLU Chatbot
Đo lường:  Chất lượng RAG, Hiệu năng, Retrieval Quality, Tài nguyên
"""

import json
import time
import os
import sys
import psutil
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict


sys.path.append(os.path.dirname(os. path.dirname(os.path. abspath(__file__))))

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core. prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import OllamaLLM, OllamaEmbeddings


from datasets import Dataset


# ============== CẤU HÌNH ==============
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen2.5:7b"
TEST_DATASET_PATH = "/mnt/48AC6E9BAC6E82F4/Dev/TLUBot/evaluation/test_dataset.json"
RESULTS_PATH = "benchmark/results"

PROMPT_TEMPLATE = """
Bạn là TLUBot - trợ lý tư vấn tuyển sinh của Trường Đại học Thủy lợi. 

NGUYÊN TẮC: 
1. Trả lời dựa trên thông tin trong phần "Thông tin tham khảo"
2. Nếu câu hỏi về quy trình/các bước, hãy liệt kê ĐẦY ĐỦ
3.  KHÔNG bịa thông tin

Thông tin tham khảo:
{context}

Câu hỏi:  {question}

Trả lời: """

# ============== DATA CLASSES ==============
@dataclass
class PerformanceMetrics:
    """Metrics hiệu năng"""
    total_latency_avg: float      # Trung bình tổng thời gian
    total_latency_p50: float      # Percentile 50
    total_latency_p95: float      # Percentile 95
    total_latency_p99: float      # Percentile 99
    retrieval_latency_avg: float  # Thời gian retrieval
    llm_latency_avg: float        # Thời gian LLM
    throughput: float             # Requests per second
    tokens_per_second: float      # Tokens per second

@dataclass
class RetrievalMetrics:
    """Metrics chất lượng retrieval"""
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float  # Mean Reciprocal Rank

@dataclass
class ResourceMetrics:
    """Metrics tài nguyên"""
    ram_usage_mb: float
    ram_peak_mb: float
    cpu_usage_percent: float
    chroma_db_size_mb:  float

@dataclass


@dataclass
class BenchmarkResult:
    """Kết quả benchmark tổng hợp"""
    timestamp: str
    config: Dict
    performance: PerformanceMetrics
    retrieval: RetrievalMetrics
    resources: ResourceMetrics
    # ragas:  RAGASMetrics
    num_test_cases: int

# ============== HELPER FUNCTIONS ==============
def get_directory_size(path: str) -> float:
    """Tính dung lượng thư mục (MB)"""
    total_size = 0
    for dirpath, dirnames, filenames in os. walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)

def percentile(data: List[float], p: int) -> float:
    """Tính percentile"""
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]

def count_tokens(text: str) -> int:
    """Đếm số tokens (ước lượng đơn giản)"""
    return len(text.split())

# ============== BENCHMARK CLASS ==============
class RAGBenchmark:
    def __init__(self):
        self.vectorstore = None
        self. rag_chain = None
        self.retriever = None
        self.llm = None
        
    def setup(self):
        """Khởi tạo các components"""
        print("🔧 Đang khởi tạo components...")
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # Vector store
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings
        )
        
        # Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # LLM
        self. llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=0.1,
            num_predict=2048,
        )
        
        # RAG Chain
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self. rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("✅ Khởi tạo hoàn tất!")
    
    def load_test_data(self) -> List[Dict]:
        """Load test dataset"""
        with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['test_cases']
    
    def benchmark_performance(self, test_cases: List[Dict]) -> Tuple[PerformanceMetrics, List[Dict]]:
        """Benchmark hiệu năng"""
        print("\n⏱️ Đang benchmark hiệu năng...")
        
        total_latencies = []
        retrieval_latencies = []
        llm_latencies = []
        total_tokens = 0
        results = []
        
        for i, test_case in enumerate(test_cases):
            question = test_case['question']
            print(f"  [{i+1}/{len(test_cases)}] {question[: 40]}...")
            
            # Đo retrieval time
            start_retrieval = time.time()
            docs = self.retriever.invoke(question)
            retrieval_time = time.time() - start_retrieval
            retrieval_latencies.append(retrieval_time)
            
            # Đo LLM time
            start_llm = time.time()
            answer = self. rag_chain.invoke(question)
            llm_time = time.time() - start_llm - retrieval_time
            llm_latencies.append(llm_time)
            
            # Tổng thời gian
            total_time = retrieval_time + llm_time
            total_latencies.append(total_time)
            
            # Đếm tokens
            total_tokens += count_tokens(answer)
            
            results.append({
                'question':  question,
                'answer': answer,
                'contexts': [doc.page_content for doc in docs],
                'ground_truth': test_case['ground_truth'],
                'retrieval_time': retrieval_time,
                'llm_time':  llm_time,
                'total_time': total_time
            })
        
        # Tính metrics
        total_time_all = sum(total_latencies)
        
        metrics = PerformanceMetrics(
            total_latency_avg=statistics.mean(total_latencies),
            total_latency_p50=percentile(total_latencies, 50),
            total_latency_p95=percentile(total_latencies, 95),
            total_latency_p99=percentile(total_latencies, 99),
            retrieval_latency_avg=statistics.mean(retrieval_latencies),
            llm_latency_avg=statistics.mean(llm_latencies),
            throughput=len(test_cases) / total_time_all,
            tokens_per_second=total_tokens / total_time_all
        )
        
        return metrics, results
    
    def benchmark_retrieval(self, test_cases: List[Dict]) -> RetrievalMetrics:
        """Benchmark chất lượng retrieval"""
        print("\n🔍 Đang benchmark retrieval quality...")
        
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_ranks = []
        
        for test_case in test_cases: 
            question = test_case['question']
            ground_truth = test_case['ground_truth']. lower()
            
            # Lấy top 5 documents
            docs = self.retriever.invoke(question)
            
            # Kiểm tra hit
            found_rank = None
            for rank, doc in enumerate(docs, 1):
                # Kiểm tra xem ground truth có overlap với retrieved doc không
                doc_content = doc.page_content.lower()
                
                # Tính overlap đơn giản
                gt_words = set(ground_truth.split())
                doc_words = set(doc_content.split())
                overlap = len(gt_words & doc_words) / len(gt_words) if gt_words else 0
                
                if overlap > 0.3:  # Threshold 30% overlap
                    if found_rank is None:
                        found_rank = rank
                    if rank == 1:
                        hits_at_1 += 1
                    if rank <= 3:
                        hits_at_3 += 1
                    if rank <= 5:
                        hits_at_5 += 1
                    break
            
            # Tính Reciprocal Rank
            if found_rank: 
                reciprocal_ranks. append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        n = len(test_cases)
        metrics = RetrievalMetrics(
            hit_rate_at_1=hits_at_1 / n,
            hit_rate_at_3=hits_at_3 / n,
            hit_rate_at_5=hits_at_5 / n,
            mrr=statistics. mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        )
        
        return metrics
    
    def benchmark_resources(self) -> ResourceMetrics:
        """Benchmark tài nguyên"""
        print("\n💻 Đang đo tài nguyên...")
        
        process = psutil.Process()
        
        # RAM
        memory_info = process.memory_info()
        ram_usage = memory_info.rss / (1024 * 1024)
        
        # CPU
        cpu_percent = process.cpu_percent(interval=1.0)
        
        # ChromaDB size
        chroma_size = get_directory_size(CHROMA_PATH) if os.path.exists(CHROMA_PATH) else 0
        
        metrics = ResourceMetrics(
            ram_usage_mb=ram_usage,
            ram_peak_mb=ram_usage,  # Có thể track peak riêng
            cpu_usage_percent=cpu_percent,
            chroma_db_size_mb=chroma_size
        )
        
        return metrics
    
    # def benchmark_ragas(self, results: List[Dict]) -> RAGASMetrics:
    #     """Benchmark với RAGAS sử dụng Ollama"""
    #     print("\n📊 Đang đánh giá với RAGAS (sử dụng Ollama)...")
        
    #     # ============== CẤU HÌNH OLLAMA CHO RAGAS ==============
    #     # Khởi tạo LLM Ollama
    #     ollama_llm = OllamaLLM(
    #         model="qwen2.5:7b",  # Hoặc model bạn đang dùng
    #         temperature=0.1,
    #     )
        
    #     # Khởi tạo Embeddings Ollama (hoặc dùng HuggingFace)
    #     # Cách 1: Dùng Ollama embeddings
    #     # ollama_embeddings = OllamaEmbeddings(model="qwen2.5:7b")
        
    #     # Cách 2: Dùng HuggingFace embeddings (khuyến nghị - nhanh hơn)
    #     from langchain_community.embeddings import HuggingFaceEmbeddings
    #     hf_embeddings = HuggingFaceEmbeddings(
    #         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #         model_kwargs={'device': 'cpu'}
    #     )
        
    #     # Wrap cho RAGAS
    #     ragas_llm = LangchainLLMWrapper(ollama_llm)
    #     ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        
    #     # ============== PREPARE DATASET ==============
    #     data = {
    #         'user_input': [r['question'] for r in results],
    #         'response': [r['answer'] for r in results],
    #         'retrieved_contexts': [r['contexts'] for r in results],
    #         'reference':  [r['ground_truth'] for r in results]
    #     }
    #     dataset = Dataset.from_dict(data)
        
    #     # ============== EVALUATE ==============
    #     ragas_result = evaluate(
    #         dataset=dataset,
    #         metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    #         llm=ragas_llm,
    #         embeddings=ragas_embeddings,
    #     )
        
    #     metrics = RAGASMetrics(
    #         faithfulness=float(ragas_result['faithfulness']),
    #         answer_relevancy=float(ragas_result['answer_relevancy']),
    #         context_precision=float(ragas_result['context_precision']),
    #         context_recall=float(ragas_result['context_recall'])
    #     )
        
    #     return metrics
    
    def run_full_benchmark(self) -> BenchmarkResult:
        """Chạy benchmark toàn diện"""
        print("🚀 BẮT ĐẦU BENCHMARK TOÀN DIỆN")
        print("=" * 60)
        
        # Setup
        self.setup()
        
        # Load test data
        test_cases = self.load_test_data()
        print(f"\n📋 Số test cases: {len(test_cases)}")
        
        # Benchmark Performance
        perf_metrics, results = self. benchmark_performance(test_cases)
        
        # Benchmark Retrieval
        retrieval_metrics = self.benchmark_retrieval(test_cases)
        
        # Benchmark Resources
        resource_metrics = self.benchmark_resources()
        
        # Benchmark RAGAS
        # ragas_metrics = self.benchmark_ragas(results)
        
        # Tổng hợp kết quả
        benchmark_result = BenchmarkResult(
            timestamp=datetime.now().isoformat(),
            config={
                'embedding_model': EMBEDDING_MODEL,
                'llm_model':  LLM_MODEL,
                'chroma_path': CHROMA_PATH,
                'retriever_k': 5
            },
            performance=perf_metrics,
            retrieval=retrieval_metrics,
            resources=resource_metrics,
            # ragas=ragas_metrics,
            num_test_cases=len(test_cases)
        )
        
        return benchmark_result

def print_benchmark_results(result: BenchmarkResult):
    """In kết quả benchmark đẹp"""
    print("\n" + "=" * 70)
    print("📊 KẾT QUẢ BENCHMARK TOÀN DIỆN")
    print("=" * 70)
    
    # Performance
    print("\n⏱️  HIỆU NĂNG (PERFORMANCE)")
    print("-" * 50)
    print(f"{'Metric':<35} {'Value':<15} {'Target':<15}")
    print("-" * 50)
    print(f"{'Total Latency (avg)':<35} {result.performance.total_latency_avg:.3f}s       < 5s")
    print(f"{'Total Latency (P50)':<35} {result.performance.total_latency_p50:.3f}s")
    print(f"{'Total Latency (P95)':<35} {result.performance.total_latency_p95:.3f}s")
    print(f"{'Total Latency (P99)':<35} {result.performance.total_latency_p99:.3f}s")
    print(f"{'Retrieval Latency (avg)':<35} {result.performance. retrieval_latency_avg:.3f}s       < 0.5s")
    print(f"{'LLM Latency (avg)':<35} {result.performance. llm_latency_avg:.3f}s       < 4s")
    print(f"{'Throughput':<35} {result.performance.throughput:.3f} req/s   > 1 req/s")
    print(f"{'Tokens per Second':<35} {result. performance.tokens_per_second:.1f} tok/s   > 20 tok/s")
    
    # Retrieval
    print("\n🔍 CHẤT LƯỢNG RETRIEVAL")
    print("-" * 50)
    print(f"{'Hit Rate @1':<35} {result. retrieval.hit_rate_at_1:.4f}        > 0.60")
    print(f"{'Hit Rate @3':<35} {result.retrieval.hit_rate_at_3:.4f}        > 0.75")
    print(f"{'Hit Rate @5':<35} {result.retrieval.hit_rate_at_5:.4f}        > 0.80")
    print(f"{'MRR (Mean Reciprocal Rank)':<35} {result.retrieval.mrr:.4f}        > 0.70")
    
    # RAGAS
    # print("\n📈 CHẤT LƯỢNG RAG (RAGAS)")
    # print("-" * 50)
    # perf_icon = lambda x, t: "✅" if x >= t else "❌"
    # print(f"{'Faithfulness':<35} {result.ragas.faithfulness:.4f}        {perf_icon(result. ragas.faithfulness, 0.85)} > 0.85")
    # print(f"{'Answer Relevancy':<35} {result.ragas.answer_relevancy:.4f}        {perf_icon(result.ragas.answer_relevancy, 0.70)} > 0.70")
    # print(f"{'Context Precision':<35} {result. ragas.context_precision:.4f}        {perf_icon(result.ragas.context_precision, 0.70)} > 0.70")
    # print(f"{'Context Recall':<35} {result. ragas.context_recall:.4f}        {perf_icon(result.ragas.context_recall, 0.80)} > 0.80")
    
    # Resources
    print("\n💻 TÀI NGUYÊN (RESOURCES)")
    print("-" * 50)
    print(f"{'RAM Usage':<35} {result.resources. ram_usage_mb:.1f} MB")
    print(f"{'CPU Usage':<35} {result. resources.cpu_usage_percent:.1f}%")
    print(f"{'ChromaDB Size':<35} {result.resources.chroma_db_size_mb:.1f} MB")
    
    # Overall Score
    # print("\n" + "=" * 70)
    # overall_score = (
    #     result.ragas.faithfulness * 0.25 +
    #     result.ragas.answer_relevancy * 0.25 +
    #     result. ragas.context_precision * 0.25 +
    #     result.ragas.context_recall * 0.25
    # )
    # print(f"🏆 ĐIỂM TỔNG HỢP (RAGAS): {overall_score:.4f}")
    # print("=" * 70)

def save_benchmark_results(result: BenchmarkResult):
    """Lưu kết quả benchmark"""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_PATH, f"benchmark_{timestamp}. json")
    
    # Convert to dict
    result_dict = {
        'timestamp': result.timestamp,
        'config':  result.config,
        'performance': asdict(result.performance),
        'retrieval': asdict(result.retrieval),
        'resources': asdict(result.resources),
        # 'ragas': asdict(result.ragas),
        'num_test_cases': result.num_test_cases
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu kết quả:  {filename}")

def main():
    benchmark = RAGBenchmark()
    result = benchmark.run_full_benchmark()
    
    print_benchmark_results(result)
    # save_benchmark_results(result)
    
    print("\n🎉 BENCHMARK HOÀN TẤT!")

if __name__ == "__main__":
    main()