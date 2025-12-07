import time
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
from docling.datamodel.base_models import InputFormat

# 1. Cấu hình các tùy chọn cho Pipeline
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True  # Bắt buộc dùng OCR để nhận diện mặt chữ
pipeline_options.do_table_structure = True # Giữ cấu trúc bảng

# 2. Cấu hình ngôn ngữ OCR là Tiếng Việt (vie) và Tiếng Anh (eng)
# Lưu ý: Máy bạn cần cài Tesseract OCR và gói ngôn ngữ tiếng Việt
# Nếu dùng EasyOCR (mặc định của một số bản docling), nó sẽ tự tải model
pipeline_options.ocr_options = TesseractOcrOptions(lang=["vie", "eng"])

# 3. Khởi tạo Converter với cấu hình trên
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# --- 2. THIẾT LẬP THƯ MỤC ---
# Đường dẫn thư mục chứa file PDF đầu vào
input_dir = Path("/mnt/48AC6E9BAC6E82F4/Dev/TLUBot/data/raw") 
# Đường dẫn thư mục chứa kết quả đầu ra
output_dir = Path("/mnt/48AC6E9BAC6E82F4/Dev/TLUBot/data/processed") 


# Lấy danh sách tất cả file .pdf trong thư mục input
pdf_files = list(input_dir.glob("*.pdf"))

if not pdf_files:
    print(f"⚠️ Không tìm thấy file PDF nào trong thư mục '{input_dir}'!")
    print("Vui lòng tạo thư mục và copy file PDF vào đó.")
else:
    print(f"📂 Tìm thấy {len(pdf_files)} file PDF. Bắt đầu xử lý hàng loạt...\n")
    
    start_time = time.time()

    # --- 3. VÒNG LẶP XỬ LÝ TỪNG FILE ---
    for index, pdf_file in enumerate(pdf_files, 1):
        try:
            print(f"[{index}/{len(pdf_files)}] Đang xử lý: {pdf_file.name} ...")
            
            # Chuyển đổi
            result = converter.convert(pdf_file)
            markdown_output = result.document.export_to_markdown()
            
            # Tạo tên file output (giữ tên cũ, thay đuôi .pdf bằng .md)
            output_filename = pdf_file.stem + ".md"
            output_path = output_dir / output_filename
            
            # Lưu file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_output)
                
            print(f"   ✅ Đã xong! Lưu tại: {output_path}")
            
        except Exception as e:
            # Nếu có lỗi ở 1 file nào đó, in lỗi và tiếp tục file tiếp theo
            print(f"   ❌ Lỗi khi xử lý file {pdf_file.name}: {e}")

    total_time = time.time() - start_time
    print(f"\n🎉 Hoàn tất quá trình! Tổng thời gian: {total_time:.2f} giây.")