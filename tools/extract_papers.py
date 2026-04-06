#!/usr/bin/env python3
import os
import sys

try:
    from pypdf import PdfReader
except Exception:
    print("pypdf not installed. Please run: python -m pip install --user pypdf")
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPERS_DIR = os.path.join(ROOT, 'Papers')
OUT_DIR = os.path.join(ROOT, 'preprocessed_data', 'extracted_papers')
os.makedirs(OUT_DIR, exist_ok=True)

pdfs = sys.argv[1:]
if not pdfs:
    pdfs = [
        'Project_Outline__Hybrid_Codes_for_Amplitude_Damping_Channels.pdf',
        'Jackson et al. - 2016 - Codeword stabilized quantum codes for asymmetric c.pdf'
    ]

for pdf_name in pdfs:
    pdf_path = os.path.join(PAPERS_DIR, pdf_name)
    if not os.path.exists(pdf_path):
        print(f"Missing: {pdf_path}")
        continue
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print(f"Failed to open {pdf_name}: {e}")
        continue
    text = []
    pages = min(len(reader.pages), 20)
    for i in range(pages):
        page = reader.pages[i]
        page_text = page.extract_text() or ''
        text.append(page_text)
    out_name = pdf_name.replace('.pdf', '.txt').replace(' ', '_')
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(text))
    print(f"Wrote: {out_path}")

print('Done.')
