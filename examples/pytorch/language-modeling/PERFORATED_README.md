# PerforatedAI MLM Benchmark

Baseline commands for comparing `google/bert_uncased_L-2_H-128_A-2` across three datasets of increasing difficulty. All three use held-out splits so train/test/validation are drawn from non-overlapping sources — improving training perplexity reflects genuine generalization, not memorization.

`bert-small` (4 layers, 512 hidden dims) is expected to outperform `bert-tiny` (2 layers, 128 hidden dims) on all three. The gap widens as dataset complexity increases.

Model name shortcuts accepted by `--model_name_or_path`: `tiny`, `mini`, `small`, `medium` (expand to the corresponding `google/bert_uncased_L-*` checkpoints).

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

## Notes

- `--max_train_samples` / `--max_eval_samples` on FineWeb-Edu keep run time reasonable; the dataset is large enough that 500K samples still far exceeds what either model can memorize.
- To run the same commands for `bert-small` comparisons, replace `tiny` with `small` and update `--output_dir` accordingly.
- Eval metric reported is perplexity (lower is better), derived from the masked language modeling cross-entropy loss.
- Available model shortcuts: `tiny` (L-2/H-128), `mini` (L-4/H-256), `small` (L-4/H-512), `medium` (L-8/H-512).
