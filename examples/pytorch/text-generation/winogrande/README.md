# Winogrande CLM fine-tuning setup

This setup formats Winogrande as a causal language modeling (CLM) task so you can fine-tune Qwen 0.5B or 1.5B for generation experiments.

## 1) Prepare data

Use the helper script to write JSONL files with a `text` field that `run_clm_no_trainer.py` can ingest.

- Train always includes the correct answer.
- Validation can optionally include the answer (for perplexity), or exclude it (for prompt-only eval).

## 2) Fine-tune

Use the existing CLM script in this repo:

- Script: [examples/pytorch/language-modeling/run_clm_no_trainer.py](../../language-modeling/run_clm_no_trainer.py)

Recommended model IDs:
- Qwen/Qwen1.5-0.5B
- Qwen/Qwen1.5-1.8B

If your model requires remote code, pass `--trust_remote_code`.

## 3) Prompt format

Each training example is formatted like:

Sentence: <sentence>
Option1: <option1>
Option2: <option2>
Answer: <correct option text>

At generation time, provide the prompt up to `Answer:` and let the model complete.

## 4) Suggested arguments

For small GPUs, start with:

- `--block_size 256`
- `--per_device_train_batch_size 2`
- `--gradient_accumulation_steps 8`
- `--learning_rate 2e-5`
- `--num_train_epochs 3`

Adjust for your hardware.

## 5) Ada6000 quick start (Qwen 1.5B)

Prepare data:

python examples/pytorch/text-generation/winogrande/prepare_winogrande_clm.py \
	--output_dir /tmp/winogrande_clm

Train:

	python run_clm_no_trainer.py \
	--model_name_or_path Qwen/Qwen1.5-1.8B \
	--dataset_name json \
	--train_file winogrande/train.json \
	--validation_file winogrande/validation.json \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--learning_rate 2e-5 \
	--num_train_epochs 3 \
	--block_size 256 \
	--output_dir /tmp/qwen15-1p8b-winogrande \
	--trust_remote_code

These settings target a single Ada6000 (48GB) and should fit comfortably. Increase batch size or block size if you want higher throughput.
