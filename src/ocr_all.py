import os
from pathlib import Path
import subprocess

RAW_DIR = Path("data/raw")
OCR_DIR = Path("data/ocr")
OCR_DIR.mkdir(parents=True, exist_ok=True)

def ocr_pdf(input_path, output_path):
    cmd = [
        "ocrmypdf",
        "-l", "vie+eng",       # OCR tiếng Việt + tiếng Anh
        "--skip-text",
        "--output-type", "pdf",
        str(input_path),
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

def main():
    for pdf_file in RAW_DIR.glob("*.pdf"):
        out_file = OCR_DIR / pdf_file.name.replace(".pdf", "_ocr.pdf")
        print(f"OCR: {pdf_file} → {out_file}")
        ocr_pdf(pdf_file, out_file)

if __name__ == "__main__":
    main()
