# Resuming the BERT dendrite experiments after the 2026-10-11 reboot

Four runs were stopped at 16:27 on 2026-10-11. Their partial logs are kept as `*_INTERRUPTED.log` in the repo root.
PAI can't resume mid-run, so the dendrite runs restart from scratch. Training is deterministic, so the reruns should
reproduce the completed cycles. Run everything from the repo root:

```bash
cd ~/PAI/PerforatedAI/transformers-perforated
ARGS=pai_experiment_scripts/pai_common_args.txt
SCRIPT=examples/pytorch/language-modeling/run_mlm_perforated.py
```

## 1. CNN/DailyMail stage-1 baseline (continue from its best checkpoint)

It ran 16 cycles (best eval loss 2.427910, last gain 0.13% per cycle) and wasn't converged yet. Its best weights are
exported to `output/bert-mini-cnn_dailymail-baseline-partial-cycle16`. Continuing from there runs the same LR restarts
from the same weights, so the plateau check simply picks up where it left off.

```bash
CUDA_VISIBLE_DEVICES=0 nohup ENV/bin/python $SCRIPT --model_name_or_path ./output/bert-mini-cnn_dailymail-baseline-partial-cycle16 \
  --tokenizer_name google/bert_uncased_L-4_H-256_A-4 --dataset_name abisee/cnn_dailymail --dataset_config_name 3.0.0 \
  --text_column_name article --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 2e-5 \
  --num_train_epochs 3 --eval_strategy epoch --max_seq_length 128 --do_train --do_eval --pai_stage baseline \
  --output_dir ./output/bert-mini-cnn_dailymail-baseline < /dev/null > mini_cnn_baseline.log 2>&1 &
```

Once it saves `./output/bert-mini-cnn_dailymail-baseline`, launch the matched pair: a restart-only control
(`--pai_stage baseline --pai_improvement_threshold 0 --pai_max_dendrites 3`) and wide ×30 dendrites
(`--pai_stage dendrites --pai_config_file pai_mini_targets_wide.json --pai_dendrite_lr_multiplier 30
--pai_improvement_threshold 0 --pai_max_dendrites 3`). Use the same data arguments as above, and
`--model_name_or_path ./output/bert-mini-cnn_dailymail-baseline`.

## 2. Mini FineWeb-Edu dendrite runs (restart from scratch)

| Run | Progress when stopped | Command |
|---|---|---|
| Wide, top layer only, ×100, 3 dendrites | cycle 1 of 3 | `CUDA_VISIBLE_DEVICES=0 nohup ENV/bin/python $SCRIPT $(cat $ARGS) --pai_stage dendrites --pai_config_file pai_mini_targets_wide_top1.json --pai_dendrite_lr_multiplier 100 --output_dir ./output/bert-mini-fineweb-edu-dendrites-wide-top1-dlr100 < /dev/null > mini_fineweb_dendrites_wide_top1_dlr100.log 2>&1 &` |
| Wide, top 2 layers, ×30, 6 dendrites | cycle 4 of 6 (−0.0299 vs control) | `CUDA_VISIBLE_DEVICES=1 nohup ENV/bin/python $SCRIPT $(sed 's/--pai_max_dendrites 3/--pai_max_dendrites 6/' $ARGS) --pai_stage dendrites --pai_config_file pai_mini_targets_wide_top2.json --pai_dendrite_lr_multiplier 30 --output_dir ./output/bert-mini-fineweb-edu-dendrites-wide-top2-dlr30-d6 < /dev/null > mini_fineweb_dendrites_wide_top2_dlr30_d6.log 2>&1 &` |
| Attention, top 2 layers, ×30, 6 dendrites, from pretrained (no stage 1) | cycle 0 of 6 | `CUDA_VISIBLE_DEVICES=1 nohup ENV/bin/python $SCRIPT $(sed -e 's#--model_name_or_path ./output/bert-mini-fineweb-edu-baseline --tokenizer_name google/bert_uncased_L-4_H-256_A-4#--model_name_or_path mini#' -e 's/--pai_max_dendrites 3/--pai_max_dendrites 6/' $ARGS) --pai_stage dendrites --pai_config_file pai_mini_targets_attn_top2.json --pai_dendrite_lr_multiplier 30 --output_dir ./output/bert-mini-fineweb-edu-dendrites-from-pretrained-attn-top2-dlr30 < /dev/null > mini_fineweb_dendrites_from_pretrained_attn_top2_dlr30.log 2>&1 &` |

## Results report

`ENV/bin/python pai_experiment_scripts/report.py` regenerates `pai_bert_dendrite_experiments.md` from the logs. To
add a new run, add a `(label, log name)` entry to `EXPERIMENTS` in `report.py`.
