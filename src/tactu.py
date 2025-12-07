import re
import json

raw_text = """
 VIỆN NGHIÊN CỨU ỨNG DỤNG CÔNG NGHỆ VÀ HỢP TÁC DOANH NGHIỆP

Điện thoại: 0964.738.089

E-mail: iartep@tlu.edu.vn, iartep.tlu@gmail.com

Website: https://www.iartep.com/

Fanpage: https://www.facebook.com/iartep.tlu

Trụ sở: 175 Tây Sơn, Phường Kim Liên, Tp. Hà Nội
Chức năng nhiệm vụ:
Hợp tác doanh nghiệp
    Thiết lập quan hệ hợp tác với các cơ sở giáo dục, doanh nghiệp trong và ngoài nước trong các lĩnh vực Giáo dục – đào tạo, Tuyển dụng – phát triển nguồn nhân lực chất lượng cao.
    Tư vấn tuyển sinh, tổ chức các chương trình liên kết đào tạo, định hướng nghề nghiệp và việc làm cho sinh viên.
Khoa học công nghệ
    Nghiên cứu, đào tạo chuyên sâu, chuyển giao công nghệ trong các lĩnh vực mũi nhọn: AI, robot, cơ khí – điện – điện tử, công nghệ chế tạo.
    Thúc đẩy khởi nghiệp, đổi mới sáng tạo trong các lĩnh vực: công nghiệp, nông nghiệp thông minh, tài nguyên – năng lượng tái tạo, môi trường.

Bộ máy tổ chức:

Viện có tư cách pháp nhân, có con dấu và tài khoản riêng. Bộ máy tổ chức của Viện được vận hành dưới sự quản lý chung của Trường Đại học Thủy Lợi, bao gồm Văn phòng Viện và 5 trung tâm: Trung tâm Liên kết Đào tạo và Tư vấn Du học, Trung tâm Hợp tác Doanh nghiệp, Trung tâm Nghiên cứu Ứng dụng Công nghệ – Tự động hóa, Trung tâm Đào tạo và Chuyển giao Công nghệ, Trung tâm Hỗ trợ Đổi mới Sáng tạo và Khởi nghiệp.

Viện hiện có 12 cán bộ nhân viên trong đó có 4 tiến sĩ, 04 thạc sĩ và 04 cử nhân. Ngoài ra, Viện có hơn 30 cán bộ giảng viên kiêm nhiệm thuộc trường Đại học Thủy Lợi và cán bộ thuộc các đơn vị ngoài trường.
"""

def parse_organization_info(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Giả định dòng đầu tiên luôn là Tên đơn vị (Entity Name)
    entity_name = lines[0] 
    
    structured_data = []
    current_category = "Thông tin chung" # Ví dụ: Liên hệ, Chi nhánh, Chức năng...
    
    for line in lines[1:]:
        # 1. Phát hiện các mục con (Category)
        if line.endswith(":") and len(line) < 50:
            current_category = line.replace(":", "")
            continue
            
        # 2. Tạo nội dung đã được làm giàu (Enriched Content)
        # Đây là bước quan trọng nhất: Ghép tên đơn vị vào nội dung
        
        # Nếu dòng chứa thông tin chi nhánh, ta giữ nguyên ngữ cảnh chi nhánh
        if line.startswith("–") or line.startswith("-"):
            contextual_text = f"{entity_name} - {current_category}: {line.replace('–', '').strip()}"
        else:
            contextual_text = f"Thông tin của {entity_name} ({current_category}): {line}"

        # 3. Tạo record
        record = {
            "content_original": line,
            "content_for_embedding": contextual_text, # RAG sẽ dùng dòng này để tìm kiếm
            "metadata": {
                "entity": entity_name,
                "category": current_category
            }
        }
        structured_data.append(record)
        
    return structured_data

# Chạy thử
data = parse_organization_info(raw_text)

# In kết quả minh họa
print(f"--- Dữ liệu RAG sẽ đọc (content_for_embedding) ---")
for item in data:
    print(item["content_for_embedding"])