# evaluation/evaluate_ragas.py
"""
Đánh giá chatbot TLU với RAGAS framework
Sử dụng Ollama (local LLM) thay vì OpenAI
"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path. dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas. metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas. llms import LangchainLLMWrapper
from ragas. embeddings import LangchainEmbeddingsWrapper
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cấu hình
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEST_DATASET_PATH = "/mnt/48AC6E9BAC6E82F4/Dev/TLUBot/evaluation/test_dataset.json"
RESULTS_PATH = "evaluation/results"
OLLAMA_MODEL = "qwen2.5:7b"

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
    """Load ChromaDB"""
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
    """Tạo RAG chain"""
    llm = OllamaLLM(model=OLLAMA_MODEL)
    retriever = vectorstore. as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def load_test_dataset():
    """Load test dataset từ file JSON"""
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']


def generate_answers(rag_chain, retriever, test_cases):
    """Generate câu trả lời và lấy context cho mỗi test case"""
    results = []

    print(f"\n📝 Đang generate câu trả lời cho {len(test_cases)} test cases...")

    for i, test_case in enumerate(test_cases):
        question = test_case['question']
        ground_truth = test_case['ground_truth']

        print(f"  [{i+1}/{len(test_cases)}] {question[:50]}...")

        # Lấy context từ retriever
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generate answer
        answer = rag_chain.invoke(question)

        results.append({
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'category': test_case. get('category', 'unknown')
        })

    return results


def prepare_ragas_dataset(results):
    """Chuẩn bị dataset theo format RAGAS"""
    data = {
        'question': [r['question'] for r in results],
        'answer': [r['answer'] for r in results],
        'contexts': [r['contexts'] for r in results],
        'ground_truth': [r['ground_truth'] for r in results]
    }
    return Dataset.from_dict(data)


def setup_ragas_with_ollama():
    """Cấu hình RAGAS sử dụng Ollama thay vì OpenAI"""
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return ragas_llm, ragas_embeddings


def run_ragas_evaluation(dataset):
    """Chạy đánh giá RAGAS với Ollama"""
    print("\n🔍 Đang đánh giá với RAGAS (sử dụng Ollama)...")
    print("⏳ Quá trình này có thể mất vài phút...")

    ragas_llm, ragas_embeddings = setup_ragas_with_ollama()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return result


def extract_scores(ragas_result):
    """Trích xuất scores từ EvaluationResult object"""
    scores = {}
    metric_names = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

    # Sử dụng to_pandas() vì đã xác nhận hoạt động
    try:
        df = ragas_result.to_pandas()
        print(f"\n📊 DataFrame columns: {df.columns.tolist()}")
        
        for metric in metric_names:
            if metric in df.columns:
                # Lấy giá trị mean, bỏ qua NaN
                values = df[metric]. dropna()
                if len(values) > 0:
                    scores[metric] = float(values.mean())
                else:
                    scores[metric] = 0.0
        
        print(f"✅ Extracted scores: {scores}")
        
    except Exception as e:
        print(f"❌ Error extracting scores: {e}")
        # Fallback: trả về scores rỗng
        scores = {metric: 0.0 for metric in metric_names}

    return scores


def save_results(ragas_scores, generated_answers):
    """Lưu kết quả đánh giá"""
    os.makedirs(RESULTS_PATH, exist_ok=True)

    timestamp = datetime.now(). strftime("%Y%m%d_%H%M%S")
    scores_file = os. path.join(RESULTS_PATH, f"ragas_scores_{timestamp}.json")

    # Convert scores to serializable format
    serializable_scores = {}
    for k, v in ragas_scores.items():
        try:
            if v is not None:
                serializable_scores[k] = round(float(v), 4)
            else:
                serializable_scores[k] = None
        except Exception:
            serializable_scores[k] = str(v)

    scores_data = {
        'timestamp': timestamp,
        'overall_scores': serializable_scores,
        'num_test_cases': len(generated_answers),
        'detailed_results': generated_answers
    }

    with open(scores_file, 'w', encoding='utf-8') as f:
        json. dump(scores_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n💾 Đã lưu kết quả tại: {scores_file}")
    return scores_file


def print_results(ragas_scores):
    """In kết quả đánh giá"""
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ RAGAS (với Ollama)")
    print("=" * 60)

    if not ragas_scores:
        print("\n⚠️ Không có scores để hiển thị!")
        print("💡 Thử chạy evaluate_simple.py thay thế")
        return

    print(f"\n{'Metric':<25} {'Score':<10} {'Đánh giá':<20}")
    print("-" * 55)

    metrics_info = {
        'faithfulness': 'Độ trung thực',
        'answer_relevancy': 'Độ liên quan',
        'context_precision': 'Độ chính xác context',
        'context_recall': 'Độ đầy đủ context',
    }

    total_score = 0.0
    count = 0

    for metric, name in metrics_info. items():
        if metric in ragas_scores:
            score = ragas_scores[metric]
            
            # Chuyển đổi score sang float an toàn
            try:
                if score is None:
                    score_val = 0.0
                else:
                    score_val = float(score)
            except (ValueError, TypeError):
                score_val = 0.0

            # Đánh giá mức độ
            if score_val >= 0.8:
                rating = "✅ Tốt"
            elif score_val >= 0.6:
                rating = "⚠️ Khá"
            else:
                rating = "❌ Cần cải thiện"

            # In kết quả - FIX: không có khoảng trắng trong format specifier
            print(f"{name:<25} {score_val:. 4f}     {rating}")
            
            total_score += score_val
            count += 1

    # Tính điểm trung bình
    if count > 0:
        avg_score = total_score / count
        print("-" * 55)
        print(f"{'ĐIỂM TRUNG BÌNH':<25} {avg_score:.4f}")

    print("=" * 60)


def main():
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ CHATBOT VỚI RAGAS")
    print("📌 Sử dụng Ollama (Local LLM) thay vì OpenAI")
    print("=" * 60)

    # 1. Load components
    print("\n📚 Đang load vector store...")
    vectorstore = load_vectorstore()
    rag_chain, retriever = create_rag_chain(vectorstore)

    # 2. Load test dataset
    print("📋 Đang load test dataset...")
    test_cases = load_test_dataset()
    print(f"   Số lượng test cases: {len(test_cases)}")

    # 3.  Generate answers
    generated_answers = generate_answers(rag_chain, retriever, test_cases)

    # 4.  Prepare RAGAS dataset
    print("\n🔧 Đang chuẩn bị dataset cho RAGAS...")
    ragas_dataset = prepare_ragas_dataset(generated_answers)

    # 5.  Run RAGAS evaluation
    ragas_result = run_ragas_evaluation(ragas_dataset)

    # 6. Extract scores từ result object
    print("\n📊 Đang trích xuất kết quả...")
    ragas_scores = extract_scores(ragas_result)

    # # 7. Print results
    # print_results(ragas_scores)

    # 8. Save results
    # save_results(ragas_scores, generated_answers)

    print("\n🎉 HOÀN THÀNH ĐÁNH GIÁ!")


if __name__ == "__main__":
    main()