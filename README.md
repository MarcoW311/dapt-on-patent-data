# This is the repo for 10-623 GenAI Group Project at CMU

## Code Structure
-src\
--perplexity.ipynb

Now this repo only contains baseline eval code.

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
