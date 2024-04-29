import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import os

torch.random.manual_seed(0)

files = glob.glob("data_cleaned/easyocr/*.txt")
# import re
# kv_regex = r'"([^"]+)":\s*("[^"]*"|[\d.]+),?'

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
    # use_flash_attention_2=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

chunk_size = 1000
batch_size = 2

generation_args = {
    "max_new_tokens": int(chunk_size * 1.5),
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

files = glob.glob("data_cleaned/easyocr/*.txt")
print(len(files))
# files = ['/home/ec2-user/dapt-on-patent-data/data_cleaned/easyocr/RE049791.txt']
for file in tqdm(files):
    post_filename = file.replace('easyocr', 'post')
    if os.path.exists(post_filename):
        continue

    texts = []
    with open(file, 'r') as f:
        texts.append(f.read())
    with open(file.replace("easyocr", "tesseract"), 'r') as f:
        texts.append(f.read())
    print(file.split('/')[-1], len(texts[0]))
    
    limit = min(len(texts[0]), 100_000)
    full_output = ""
    for i in tqdm(range(0, limit, chunk_size * batch_size)):
        # instruction = f"Fix the OCR output. OCR: \n{texts[1][i:i+chunk_size]}"
        instructions = [f"You are give two OCR outputs on the same text. Combine the output and fix any errors you detect. " + \
            "Provide cleaned text only and no additional response.\n" + \
            f"OCR 1: \n{texts[0][i+chunk_size*b:i+chunk_size*(b+1)]}\n\n" + \
            f"OCR 2: \n{texts[1][i+chunk_size*b:i+chunk_size*(b+1)]}.\n" for b in range(batch_size)]

        messages = [[
                {"role": "system", "content": "You are an expert at detecting OCR errors and cleaning OCR outputs."},
                {"role": "user", "content": instruction},
            ] for instruction in instructions]

        output = pipe(messages, batch_size=batch_size, **generation_args)

        full_output += '\n'.join([output[i][0]['generated_text'] for i in range(batch_size)]) + '\n'

        # output = re.findall(kv_regex, output[0]['generated_text'])[0]
        # print(output)


    with open(post_filename, 'w') as f:
        f.write(full_output)
