# This is the repo for 10-623 GenAI Group Project at CMU

## Further pretraining gpt-2
The code is in `src/run_clm.py` and I used the command below.
```bash
python run_clm.py \
    --model_name_or_path openai-community/gpt2 \
    --train_file data_cleaned/easyocr_merged.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --output_dir temp/
    --report_to wandb
```


## Phi-2 cleaning ocr output
```bash
python src/clean_ocr.py \
    --num_gpus 4 \
    --gpu 1 \
    --batch_size 1 \
    --chunk_size 20000 \
    --flash_attention
```
Only add `flash_attention` if your GPU supports flash attention. You can also experiment with adding `--long_prompt` with a slightly different few-shot prompting.
