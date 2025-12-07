# evaluation/evaluate_traditional. py
"""
Đánh giá chatbot với metrics truyền thống: BLEU, ROUGE
Dùng để so sánh và bổ sung cho RAGAS
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.dirname(os.path. dirname(os.path.abspath(__file__))))

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Download nltk data
nltk.download('punkt', quiet=True)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cấu hình
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEST_DATASET_PATH = "evaluation/test_dataset.json"
RESULTS_PATH = "evaluation/results"

PROMPT_TEMPLATE = """
Bạn là TLUBot - trợ lý tư vấn tuyển sinh của Trường Đại học Thủy lợi. 
Hãy trả lời câu hỏi dựa trên thông tin được cung cấp bên dưới. 
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
    return "\n\n". join(doc.page_content for doc in docs)

def create_rag_chain(vectorstore):
    """Tạo RAG chain"""
    llm = OllamaLLM(model="qwen2.5:7b")
    retriever = vectorstore.as_retriever(
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
    
    return rag_chain

def load_test_dataset():
    """Load test dataset"""
    with open(TEST_DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['test_cases']

def tokenize_vietnamese(text: str) -> List[str]:
    """Tokenize tiếng Việt đơn giản (split by space)"""
    return text.lower().split()

def calculate_bleu(reference: str, candidate: str) -> Dict[str, float]:
    """Tính BLEU score"""
    reference_tokens = [tokenize_vietnamese(reference)]
    candidate_tokens = tokenize_vietnamese(candidate)
    
    smoothing = SmoothingFunction(). method1
    
    # BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu_scores = {}
    for n in range(1, 5):
        weights = tuple([1.0/n] * n + [0.0] * (4-n))
        try:
            score = sentence_bleu(reference_tokens, candidate_tokens, 
                                  weights=weights, 
                                  smoothing_function=smoothing)
        except:
            score = 0.0
        bleu_scores[f'bleu_{n}'] = score
    
    return bleu_scores

def calculate_rouge(reference: str, candidate: str) -> Dict[str, float]:
    """Tính ROUGE scores"""
    scorer = rouge_scorer. RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge1_precision': scores['rouge1'].precision,
        'rouge1_recall': scores['rouge1'].recall,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure,
    }

def evaluate_single(question: str, answer: str, ground_truth: str) -> Dict:
    """Đánh giá một cặp answer/ground_truth"""
    bleu_scores = calculate_bleu(ground_truth, answer)
    rouge_scores = calculate_rouge(ground_truth, answer)
    
    return {
        'question': question,
        'answer': answer,
        'ground_truth': ground_truth,
        **bleu_scores,
        **rouge_scores
    }

def run_evaluation(rag_chain, test_cases: List[Dict]) -> List[Dict]:
    """Chạy evaluation cho tất cả test cases"""
    results = []
    
    print(f"\n📝 Đang đánh giá {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases):
        question = test_case['question']
        ground_truth = test_case['ground_truth']
        
        print(f"  [{i+1}/{len(test_cases)}] {question[:50]}...")
        
        # Generate answer
        answer = rag_chain. invoke(question)
        
        # Evaluate
        result = evaluate_single(question, answer, ground_truth)
        result['category'] = test_case.get('category', 'unknown')
        results. append(result)
    
    return results

def calculate_averages(results: List[Dict]) -> Dict[str, float]:
    """Tính điểm trung bình"""
    metrics = ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 
               'rouge1_f1', 'rouge2_f1', 'rougeL_f1']
    
    averages = {}
    for metric in metrics:
        scores = [r[metric] for r in results if metric in r]
        averages[metric] = sum(scores) / len(scores) if scores else 0.0
    
    return averages

def calculate_category_averages(results: List[Dict]) -> Dict[str, Dict]:
    """Tính điểm trung bình theo category"""
    categories = {}
    
    for result in results:
        cat = result. get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)
    
    category_averages = {}
    for cat, cat_results in categories. items():
        category_averages[cat] = calculate_averages(cat_results)
    
    return category_averages

def save_results(results: List[Dict], averages: Dict, category_averages: Dict):
    """Lưu kết quả"""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_PATH, f"traditional_scores_{timestamp}.json")
    
    data = {
        'timestamp': timestamp,
        'overall_averages': averages,
        'category_averages': category_averages,
        'num_test_cases': len(results),
        'detailed_results': results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu kết quả tại: {filename}")

def print_results(averages: Dict, category_averages: Dict):
    """In kết quả"""
    print("\n" + "="*60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ BLEU & ROUGE")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Score':<10}")
    print("-"*30)
    
    for metric, score in averages.items():
        print(f"{metric:<20} {score:. 4f}")
    
    print("\n" + "-"*60)
    print("📂 KẾT QUẢ THEO CATEGORY")
    print("-"*60)
    
    for category, scores in category_averages.items():
        print(f"\n🏷️  {category. upper()}")
        for metric, score in scores.items():
            print(f"   {metric}: {score:.4f}")
    
    print("\n" + "="*60)

def main():
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ VỚI BLEU & ROUGE")
    print("="*60)
    
    # Load components
    print("\n📚 Đang load vector store...")
    vectorstore = load_vectorstore()
    rag_chain = create_rag_chain(vectorstore)
    
    # Load test dataset
    print("📋 Đang load test dataset...")
    test_cases = load_test_dataset()
    
    # Run evaluation
    results = run_evaluation(rag_chain, test_cases)
    
    # Calculate averages
    averages = calculate_averages(results)
    category_averages = calculate_category_averages(results)
    
    # Print results
    print_results(averages, category_averages)
    
    # Save results
    save_results(results, averages, category_averages)
    
    print("\n🎉 HOÀN THÀNH ĐÁNH GIÁ!")

if __name__ == "__main__":
    main()