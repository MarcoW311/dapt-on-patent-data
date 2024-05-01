import os
import argparse
import glob
from tqdm import tqdm

def clean_ocr(
    gpu: int = 0,
    num_gpus: int = 1,
    batch_size: int = 1,
    chunk_size: int = 1000,
    long_prompt: bool = False,
    flash_attention: bool = False,
    max_len: int = 100_000,
):
    print('gpu:', gpu)
    print('num_gpus:', num_gpus)
    print('batch_size:', batch_size)
    print('chunk_size:', chunk_size)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    torch.random.manual_seed(0)

    files = glob.glob("data_cleaned/easyocr/*.txt")
    # split files according to cuda
    files = [files[i] for i in range(len(files)) if i % num_gpus == gpu]

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
        attn_implementation="flash_attention_2" if flash_attention else "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    generation_args = {
        "max_new_tokens": int(chunk_size * 3),
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    with open("prompt/example_input_short.txt", 'r') as f:
        example_input = f.read()
    with open("prompt/example_output_short.txt", 'r') as f:
        example_output = f.read()
    with open("prompt/example_input_short2.txt", 'r') as f:
        example_input2 = f.read()
    with open("prompt/example_output_short2.txt", 'r') as f:
        example_output2 = f.read()
    with open("prompt/example_input_short3.txt", 'r') as f:
        example_input3 = f.read()
    with open("prompt/example_output_short3.txt", 'r') as f:
        example_output3 = f.read()
    with open("prompt/example_input.txt", 'r') as f:
        example_input_long = f.read()
    with open("prompt/example_output.txt", 'r') as f:
        example_output_long = f.read()
    with open("prompt/prompt.txt", 'r') as f:
        instruction = f.read()
    with open('prompt/artifacts_remove.txt', 'r') as f:
        artifacts = f.read().lower().splitlines()

    instruction_sample = instruction + '\n' + (example_input_long if long_prompt else example_input) + '\n'

    print('prompt:', instruction)
    print('prompt length:', len(instruction_sample + (example_output_long if long_prompt else example_output)))

    for file in tqdm(files):
        post_filename = file.replace('easyocr', 'phi3')
        if os.path.exists(post_filename):
            continue

        texts = []
        with open(file, 'r') as f:
            texts.append(f.read())
        with open(file.replace("easyocr", "tesseract"), 'r') as f:
            texts.append(f.read())
        print(file.split('/')[-1], len(texts[0]))
        
        limit = min(len(texts[0]), len(texts[1]), max_len)
        full_output = ""

        for i in tqdm(range(0, limit, chunk_size * batch_size)):

            ocr_pairs = [(texts[0][i+chunk_size*b:i+chunk_size*(b+1)], texts[1][i+chunk_size*b:i+chunk_size*(b+1)]) for b in range(batch_size)]
            ocr_pairs = [pair for pair in ocr_pairs if len(pair[0]) > chunk_size // 2 and len(pair[1]) > chunk_size // 2]
            if len(ocr_pairs) == 0:
                continue
            inputs = [f"OCR 1:\n\n{ocr1}\n\nOCR 2:\n\n{ocr2}" for (ocr1, ocr2) in ocr_pairs]


            if long_prompt:
                messages = [[
                    {"role": "system", "content": "You are an expert at detecting OCR errors and cleaning OCR outputs."},
                    {"role": "user", "content": instruction_sample},
                    {"role": "assistant", "content": example_output_long},
                    {"role": "user", "content": example_input3},
                    {"role": "assistant", "content": example_output3},
                    {"role": "user", "content": user_input},
                ] for user_input in inputs]
            else:
                messages = [[
                        {"role": "system", "content": "You are an expert at detecting OCR errors and cleaning OCR outputs."},
                        {"role": "user", "content": instruction_sample},
                        {"role": "assistant", "content": example_output},
                        {"role": "user", "content": example_input2},
                        {"role": "assistant", "content": example_output2},
                        {"role": "user", "content": example_input3},
                        {"role": "assistant", "content": example_output3},
                        {"role": "user", "content": user_input},
                    ] for user_input in inputs]

            output = pipe(messages, batch_size=batch_size, **generation_args)

            full_output += '\n'.join([output[i][0]['generated_text'] for i in range(batch_size)]) + '\n'

        output_split = full_output.splitlines()
        filter_artifact = lambda x: any([artifact in x.lower() for artifact in artifacts]) or len(x) < 5
        filtered = [line for line in output_split if not filter_artifact(line)]
        filtered_output = '\n'.join(filtered)

        os.makedirs(os.path.dirname(post_filename), exist_ok=True)
        with open(post_filename, 'w') as f:
            f.write(filtered_output)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Clean OCR outputs using Phi-3 instruct")
    
    # Add arguments
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--gpu", type=int, default=0, help="Index of GPU to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--long_prompt", action="store_true", help="Use long prompt")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention")

    args = parser.parse_args()

    clean_ocr(
        gpu=args.gpu,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        long_prompt=args.long_prompt,
        flash_attention=args.flash_attention,
    )

if __name__ == "__main__":
    main()
