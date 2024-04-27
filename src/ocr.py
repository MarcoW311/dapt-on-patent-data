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
import xml.etree.ElementTree as ET

mode = sys.argv[1]
assert mode in ['ocrad', 'tesseract', 'easyocr']
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
    outf = f'data_cleaned/{mode}/{fname}.txt'
    if os.path.exists(outf):
        continue

    xml_file = '/'.join(pdf_file.split('/')[:-1]+[f'us-patent-image.xml'])
    metadata = ET.parse(xml_file).getroot()[0]
    portions = [(int(metadata.find(p)[0].text), int(metadata.find(p)[1].text)) for p in ['abstract-pages', 'description-pages', 'claims-pages']]
    pages = []
    for p in portions:
        pages += convert_from_path(pdf_file, first_page=p[0], last_page=min(p[0]+5, p[1]), grayscale=True)
    
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

    with open(outf, 'w') as f:
        f.write("\n".join(results))
    