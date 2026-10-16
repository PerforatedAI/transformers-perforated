# PerforatedAI MLM Benchmark

Baseline commands for comparing `google/bert_uncased_L-2_H-128_A-2` across three datasets of increasing difficulty. All three use held-out splits so train/test/validation are drawn from non-overlapping sources — improving training perplexity reflects genuine generalization, not memorization.

`bert-small` (4 layers, 512 hidden dims) is expected to outperform `bert-tiny` (2 layers, 128 hidden dims) on all three. The gap widens as dataset complexity increases.

Model name shortcuts accepted by `--model_name_or_path`: `tiny`, `mini`, `small`, `medium`, `base`, `large` (expand to the corresponding `google/bert_uncased_L-*` or standard `bert-*-uncased` checkpoints).

---

## Easy — WikiText-103

**Dataset**: `Salesforce/wikitext` / `wikitext-103-raw-v1` (~103M tokens of clean Wikipedia prose)

**Why easy**: BERT was pretrained on Wikipedia. bert-tiny's linguistic priors closely match this domain, and Wikipedia's consistent encyclopedic style limits the range of patterns to model. Expected bert-tiny eval perplexity: ~25–45. bert-small expected: ~15–28.

```bash
python examples/pytorch/language-modeling/run_mlm_perforated.py \
    --model_name_or_path tiny \
    --dataset_name Salesforce/wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --eval_strategy epoch \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --output_dir ./output/bert-tiny-wikitext103
```

---

## Medium — CNN/DailyMail

**Dataset**: `abisee/cnn_dailymail` / `3.0.0` (~300K news articles, article-level train/validation/test splits)

**Why medium**: News articles are stylistically diverse (varied topics, registers, and vocabulary) but short enough that no long-range context is required. Splits are strictly article-level with no text overlap between them. Expected bert-tiny eval perplexity: ~60–100. bert-small expected: ~35–55.

```bash
python examples/pytorch/language-modeling/run_mlm_perforated.py \
    --model_name_or_path tiny \
    --dataset_name abisee/cnn_dailymail \
    --dataset_config_name 3.0.0 \
    --text_column_name article \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --eval_strategy epoch \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --output_dir ./output/bert-tiny-cnn_dailymail
```

---

## Hard — FineWeb-Edu

**Dataset**: `HuggingFaceFW/fineweb-edu` / `sample-10BT` (~48GB cached, educational and academic web text)

**Why hard**: Filtered for educational quality — textbooks, academic writing, technical documentation. Dense vocabulary and structured argumentation far outside BERT's Wikipedia pretraining domain. Parquet-native, no legacy scripts, no trust_remote_code needed. Expected bert-tiny eval perplexity: ~150–300. bert-small expected: ~80–150 (the largest gap of the three).

```bash
python examples/pytorch/language-modeling/run_mlm_perforated.py \
    --model_name_or_path tiny \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config_name sample-10BT \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --eval_strategy epoch \
    --max_seq_length 128 \
    --max_train_samples 500000 \
    --max_eval_samples 5000 \
    --do_train \
    --do_eval \
    --output_dir ./output/bert-tiny-fineweb-edu
```

---

## Two-stage dendrite experiments

Each PAI cycle runs one LR schedule of `--num_train_epochs` epochs, and a new cycle restarts the schedule. Adding a dendrite always comes with such a restart, and a restart alone also improves a model that hasn't converged, so comparing against a model trained for a single schedule overstates what the dendrites do. Instead:

1. **Baseline** (`--pai_stage baseline`, run once): restart the schedule without adding dendrites until a cycle improves the best eval loss by less than `--pai_improvement_threshold` (default 0.1%). The best model is saved as a plain checkpoint in `--output_dir`.
2. **Dendrites** (`--pai_stage dendrites`): load that checkpoint and add dendrites. The first cycle is one more restart without dendrites, so it is the control: its gain shows what a restart alone still buys, and each dendrite cycle should be compared against it.

```bash
# Stage 1: converged baseline, saved to ./output/bert-mini-fineweb-edu-baseline
python examples/pytorch/language-modeling/run_mlm_perforated.py \
    --model_name_or_path mini \
    --dataset_name HuggingFaceFW/fineweb-edu \
    --dataset_config_name sample-10BT \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --eval_strategy epoch \
    --max_seq_length 128 \
    --max_train_samples 500000 \
    --max_eval_samples 5000 \
    --do_train \
    --do_eval \
    --pai_stage baseline \
    --output_dir ./output/bert-mini-fineweb-edu-baseline

# Stage 2: dendrites on top of the saved baseline (repeat with different perforation targets)
python examples/pytorch/language-modeling/run_mlm_perforated.py \
    --model_name_or_path ./output/bert-mini-fineweb-edu-baseline \
    --tokenizer_name google/bert_uncased_L-4_H-256_A-4 \
    ... same data and training arguments ... \
    --pai_stage dendrites \
    --pai_config_file <pai config with the perforation targets>.json \
    --output_dir ./output/bert-mini-fineweb-edu-dendrites
```

- Use the same data, eval and training arguments (including `--seed`) in both stages so the eval sets and masks are identical and the scores are directly comparable.
- Pass the original tokenizer with `--tokenizer_name` in the `dendrites` stage. The copy saved with the baseline is identical but gets a different `datasets` cache fingerprint, so the whole dataset would be tokenized again.
- `--fixed_eval_masks` (on by default) masks the eval set once. Without it, re-masking on every evaluation moves the eval loss of unchanged weights by about as much as the effects being measured.
- `--pai_switch_on_schedule_end` (on by default) starts the next cycle when the schedule reaches its end. Otherwise PAI waits `n_epochs_to_switch` epochs without improvement while the LR sits at 0.
- Without `--pai_config_file`, the `dendrites` stage looks for `<model>_<dataset>.json` in the working directory and asks for perforation targets interactively if it doesn't exist.
- `--pai_dendrite_lr_multiplier` trains dendrite parameters at a multiple of `--learning_rate`. A dendrite's output weights start at 0, so at a fine-tuning learning rate like 2e-5 they barely move within a cycle and the dendrites contribute nothing. On bert-mini / FineWeb-Edu, ×100 worked best of ×10, ×100 and ×1000.

## Notes

- `--max_train_samples` / `--max_eval_samples` on FineWeb-Edu keep run time reasonable; the dataset is large enough that 500K samples still far exceeds what either model can memorize.
- To run the same commands for `bert-small` comparisons, replace `tiny` with `small` and update `--output_dir` accordingly.
- Eval metric reported is perplexity (lower is better), derived from the masked language modeling cross-entropy loss.
- Available model shortcuts: `tiny` (L-2/H-128/A-2), `mini` (L-4/H-256/A-4), `small` (L-4/H-512/A-8), `medium` (L-8/H-512/A-8), `base` (L-12/H-768/A-12), `large` (L-24/H-1024/A-16).
