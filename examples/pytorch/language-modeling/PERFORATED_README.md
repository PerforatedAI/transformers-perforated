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

## Adding dendrites

Training runs in cycles: each cycle is one LR schedule of `--num_train_epochs` epochs, and every cycle after the first adds one dendrite and restarts the schedule. Training stops after `--pai_max_dendrites` dendrites (default 6) or at the first cycle that improves the best eval loss by less than `--pai_improvement_threshold` (default 0.1%). The defaults are the settings that worked best in our experiments, so this is all you need:

```bash
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
    --max_train_samples 2000000 \
    --max_eval_samples 5000 \
    --do_train \
    --do_eval \
    --output_dir ./output/bert-mini-fineweb-edu-dendrites
```

What the defaults do and why:

- `--pai_dendrite_top_layers 1` adds dendrites only to the top encoder layer and the MLM head's transform. Dendrites close to the output give by far the most gain per added parameter: on bert-mini, 12 dendrites in the top layer beat 6 dendrites in every layer with half the added parameters. Pass `--pai_config_file` for other placements.
- `--pai_dendrite_lr_multiplier 30` trains dendrite parameters at 30× `--learning_rate`. A dendrite's output weights start at 0, so at a fine-tuning learning rate they barely move and the dendrites add nothing.
- `--pai_switch_on_schedule_end` starts the next cycle as soon as the LR schedule ends instead of letting training continue at LR 0.
- `--fixed_eval_masks` masks the eval set once. Re-masking on every evaluation moves the eval loss of unchanged weights by about as much as a dendrite improves it, which misleads PAI's choice of the best model.

When dendrites help:

- **Give each dendrite plenty of data.** The gain grows with the amount of training per cycle: on bert-mini, 4× more training samples per epoch doubled the gain per added parameter.
- **Dendrites add capacity, so they help models that are short on capacity, not on data.** They gave bert-mini 3–4% lower eval loss and beat adding the same number of ordinary FFN neurons by about 2.4×, but gave bert-small nothing once it had enough data to keep improving on its own.

### Measuring the gain

Every new cycle also restarts the LR schedule, which improves a model that hasn't converged even without dendrites. To see what the dendrites themselves add, compare against the same number of restarts without dendrites:

1. `--pai_stage baseline` restarts the schedule without dendrites until a cycle improves the eval loss by less than `--pai_improvement_threshold`, then saves the best model as a plain checkpoint in `--output_dir`. Pass `--pai_improvement_threshold -1 --pai_max_dendrites N` instead to get exactly N restarts as a control.
2. `--pai_stage dendrites` (the default) with that checkpoint as `--model_name_or_path` adds dendrites from the same starting point. Its first cycle is one more restart without dendrites, so each later cycle can be compared with the baseline run's cycle of the same number.

- Use the same data, eval and training arguments (including `--seed`) in both runs, so the eval sets and masks are identical.
- When loading a saved checkpoint, pass the original tokenizer with `--tokenizer_name`. The saved copy is identical but gets a different `datasets` cache fingerprint, so the whole dataset would be tokenized again.

## Notes

- `--max_train_samples` / `--max_eval_samples` on FineWeb-Edu keep run time reasonable; the dataset is large enough that 500K samples still far exceeds what either model can memorize.
- To run the same commands for `bert-small` comparisons, replace `tiny` with `small` and update `--output_dir` accordingly.
- Eval metric reported is perplexity (lower is better), derived from the masked language modeling cross-entropy loss.
- Available model shortcuts: `tiny` (L-2/H-128/A-2), `mini` (L-4/H-256/A-4), `small` (L-4/H-512/A-8), `medium` (L-8/H-512/A-8), `base` (L-12/H-768/A-12), `large` (L-24/H-1024/A-16).
