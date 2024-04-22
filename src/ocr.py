from io import BytesIO
import numpy as np
import glob
import os
import tqdm
import sys
from pdf2image import convert_from_path
import easyocr
import pytesseract
from subprocess import Popen, PIPE

mode = sys.argv[1]
print(f"Mode: {mode}")

if mode == 'easyocr':
    easyocr_reader = easyocr.Reader(['en'])

pdf_path = 'data/*/*/*/*/*.pdf'
pdf_files = glob.glob(pdf_path)[:10_000]
print(len(pdf_files))

def ocrad_extract(img, encoding="iso8859-15"):
    p = Popen(["ocrad", "-"], stdin=PIPE, stdout=PIPE)
    buffer = BytesIO()
    img.save(buffer, format="ppm")

    p.stdin.write(buffer.getvalue())
    p.stdin.close()
    p.wait()
    text = p.stdout.read()
    p.stdout.close()

    return text.decode(encoding).strip("\n")

def tesseract_extract(img):
    return pytesseract.image_to_string(img, lang='eng').strip("\n")

def easyocr_extract(img):
    paragraph = easyocr_reader.readtext(np.array(img), detail = 0, paragraph=True)
    return "\n".join(paragraph)

for pdf_file in tqdm.tqdm(pdf_files):
    fname = os.path.basename(pdf_file).split('.')[0]
    pages = convert_from_path(pdf_file)
    results = []
    for page in pages:
        if mode == 'ocrad':
            output = ocrad_extract(page)
            results.append(output)
        elif mode == 'tesseract':
            output = tesseract_extract(page)
            results.append(output)
        elif mode == 'easyocr':
            output = easyocr_extract(page)
            results.append(output)

    with open(f'data_cleaned/{mode}/{fname}.txt', 'w') as f:
        f.write("\n".join(results))
    