import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

torch.random.manual_seed(0)

files = glob.glob("data_cleaned/*/11858510.txt")

texts = []
for file in files:
    with open(file, 'r') as f:
        texts.append(f.read())

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

full_output = ""
chunk_size = 1000
limit = 5_000

generation_args = {
    "max_new_tokens": int(chunk_size * 1.5),
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
for i in tqdm(range(0, limit, chunk_size)):
    # instruction = f"Fix the OCR output. OCR: \n{texts[1][i:i+chunk_size]}"
    instruction = f"You are give two OCR outputs on the same text. Combine the output and fix any errors you detect. OCR 1: \n{texts[0][i:i+chunk_size]}\n\n OCR 2: \n{texts[1][i:i+chunk_size]}. Corrected output: \n"


    messages = [
        {"role": "system", "content": "You are an expert at detecting OCR errors and cleaning OCR outputs."},
        {"role": "user", "content": instruction},
    ]

    output = pipe(messages, **generation_args)

    full_output += output[0]['generated_text']


# with open("cleaned_output.txt", 'w') as f:
#     f.write(full_output)

# with open("original_output.txt", 'w') as f:
#     f.write(texts[1])