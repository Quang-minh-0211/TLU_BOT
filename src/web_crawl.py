import requests
from bs4 import BeautifulSoup
import os

def clean_text(text):
    """Hàm làm sạch văn bản cơ bản"""
    # Xóa khoảng trắng thừa đầu đuôi và các ký tự xuống dòng
    return " ".join(text.split())

def crawl_specific_tags(url, output_folder="crawled_data"):
    try:
        # 1. Tải trang web
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Báo lỗi nếu 404/500
        
        # 2. Tạo Soup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # --- BƯỚC QUAN TRỌNG: LOẠI BỎ RÁC TRƯỚC ---
        # Xóa script, style để tránh lấy nhầm code
        for script in soup(["script", "style", "nav", "footer"]): 
            script.decompose()

        # 3. Chiến thuật lấy dữ liệu thông minh
        collected_texts = []
        
        # Lấy tất cả thẻ p và span
        tags = soup.find_all(['p', 'span'])
        
        for tag in tags:
            text = clean_text(tag.get_text())
            
            # --- BỘ LỌC CHẤT LƯỢNG (RẤT CẦN THIẾT CHO RAG) ---
            
            # Lọc 1: Bỏ qua nếu text quá ngắn (thường là icon, số trang, dấu chấm)
            if len(text) < 10: 
                continue
                
            # Lọc 2: TRÁNH TRÙNG LẶP (Logic Span trong P)
            # Nếu thẻ hiện tại là span, và cha nó là p, thì bỏ qua 
            # (vì nội dung span đã nằm trong nội dung của p rồi)
            if tag.name == 'span' and tag.parent.name == 'p':
                continue
                
            collected_texts.append(text)

        # 4. Lưu vào file
        if collected_texts:
            # Tạo tên file từ URL (để không bị trùng)
            filename = url.split("//")[-1].replace("/", "_").replace("?", "")[:50] + ".txt"
            
            # Tạo thư mục nếu chưa có
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            file_path = os.path.join(output_folder, filename)
            
            # Ghi file: Mỗi đoạn text là một dòng (tốt cho việc chunking sau này)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(collected_texts))
                
            print(f"✅ Đã lấy {len(collected_texts)} đoạn văn từ: {url}")
            print(f"   --> Lưu tại: {file_path}")
        else:
            print(f"⚠️ Không tìm thấy nội dung hợp lệ ở: {url}")

    except Exception as e:
        print(f"❌ Lỗi khi xử lý {url}: {e}")

# --- CHẠY THỬ NGHIỆM ---

# Danh sách các link bạn muốn lấy
list_urls = [
    "https://tlu.edu.vn/gioi-thieu/",
    "https://tlu.edu.vn/su-mang-muc-tieu-chien-luoc/"
]

print("Bắt đầu crawl dữ liệu thẻ <p> và <span>...")
for link in list_urls:
    crawl_specific_tags(link)