"""Write a markdown report comparing every dendrite run against its restart-only control, cycle by cycle."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import ROOT, runs  # noqa: E402

CONTROLS = {
    # Deterministic reruns of the same control, so use whichever has run the most cycles.
    "mini / FineWeb-Edu": ("mini_fineweb_restart_control6", "mini_fineweb_restart_control12"),
    "mini / WikiText-103": "mini_wikitext_restart_control3",
    "small / FineWeb-Edu": "small_fineweb_restart_control3",
    "mini / FineWeb-Edu, 2M training samples per epoch (4× longer cycles, same baseline and eval set)": "mini_fineweb2m_restart_control6",
    "small / FineWeb-Edu, 2M training samples per epoch (4× longer cycles, same baseline and eval set)": "small_fineweb2m_restart_control6",
    "mini / CNN/DailyMail": "mini_cnn_restart_control6",
    "mini / FineWeb-Edu, seed 7 (different data order, dendrite init and eval masks)": "mini_fineweb_restart_control6_seed7",
    "mini / FineWeb-Edu, from the pretrained checkpoint (no stage 1; the control is stage 1 itself)": "mini_fineweb_baseline",
}
EXPERIMENTS = {
    "mini / FineWeb-Edu": [
        ("Head only, ×1 (init_by_main; default init gives identical numbers)", "mini_fineweb_dendrites_head_initmain"),
        ("Head only, ×100", "mini_fineweb_dendrites_head_dlr100"),
        ("Head + layer3, ×1", "mini_fineweb_dendrites_head_layer3"),
        ("Wide, ×1", "mini_fineweb_dendrites_wide"),
        ("Wide, ×1, init_by_main", "mini_fineweb_dendrites_wide_initmain"),
        ("Head + layer3, ×10", "mini_fineweb_dendrites_head_layer3_dlr10"),
        ("Head + layer3, ×100", "mini_fineweb_dendrites_head_layer3_dlr100"),
        ("Wide, ×10", "mini_fineweb_dendrites_wide_dlr10"),
        ("Wide, ×30, 6 dendrites", "mini_fineweb_dendrites_wide_dlr30_d6"),
        ("No dendrites: 9606 extra FFN neurons per layer (as many parameters as 6 wide dendrites), ×30 (stopped when cycle 1 got worse)", "mini_fineweb_ffn_control_wide_dlr30_d6"),
        ("No dendrites: 9606 extra FFN neurons per layer (as many parameters as 6 wide dendrites), ×1", "mini_fineweb_ffn_control_wide_dlr1_d6"),
        ("No dendrites: 9606 extra FFN neurons per layer (as many parameters as 6 wide dendrites), ×3", "mini_fineweb_ffn_control_wide_dlr3_d6"),
        ("Wide, ×100, 6 dendrites", "mini_fineweb_dendrites_wide_dlr100_d6"),
        ("Wide, ×100, init_by_main", "mini_fineweb_dendrites_wide_initmain_dlr100"),
        ("Wide, ×300", "mini_fineweb_dendrites_wide_dlr300"),
        ("Wide, ×1000", "mini_fineweb_dendrites_wide_dlr1000"),
        ("Wide, ×100, linear (identity) dendrites", "mini_fineweb_dendrites_wide_identity_dlr100"),
        ("Wide, ×100 on dendrite output weights only", "mini_fineweb_dendrites_wide_dlr100_outputs"),
        ("Wide, ×100 on dendrite input weights only", "mini_fineweb_dendrites_wide_dlr100_weights"),
        ("Wide, dendrites at 2e-3 with main network frozen (LR 2e-7)", "mini_fineweb_dendrites_wide_frozenmain"),
        ("FFN only, ×100", "mini_fineweb_dendrites_ffn_dlr100"),
        ("Attention only, ×100, 6 dendrites", "mini_fineweb_dendrites_attn_dlr100_d6"),
        ("Attention only in top 2 layers (2-3), ×100", "mini_fineweb_dendrites_attn_top2_dlr100"),
        ("Attention only in bottom 2 layers (0-1), ×100", "mini_fineweb_dendrites_attn_bottom2_dlr100"),
        ("Attention only in top layer (3), ×100", "mini_fineweb_dendrites_attn_top1_dlr100"),
        ("Wide in top layer (3), ×100", "mini_fineweb_dendrites_wide_top1_dlr100"),
        ("Wide in top 2 layers (2-3), ×100", "mini_fineweb_dendrites_wide_top2_dlr100"),
        ("Wide in top layer (3), ×30, 6 dendrites", "mini_fineweb_dendrites_wide_top1_dlr30_d6"),
        ("Wide in top layer (3), ×30, 12 dendrites", "mini_fineweb_dendrites_wide_top1_dlr30_d12"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×30 (stopped after cycle 3)", "mini_fineweb_ffn_control_top1_dlr30_d6"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×0.3", "mini_fineweb_ffn_control_top1_dlr0.3_d6"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×1", "mini_fineweb_ffn_control_top1_dlr1_d6"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×3", "mini_fineweb_ffn_control_top1_dlr3_d6"),
        ("Wide in top 2 layers (2-3), ×30, 6 dendrites", "mini_fineweb_dendrites_wide_top2_dlr30_d6"),
        ("Attention only in top 2 layers (2-3), ×30, 6 dendrites", "mini_fineweb_dendrites_attn_top2_dlr30_d6"),
        ("Attention only in top layer (3), ×30, 6 dendrites", "mini_fineweb_dendrites_attn_top1_dlr30_d6"),
    ],
    "mini / WikiText-103": [
        ("Wide, ×100", "mini_wikitext_dendrites_wide_dlr100"),
        ("Wide, ×30", "mini_wikitext_dendrites_wide_dlr30"),
        ("Attention only, ×100", "mini_wikitext_dendrites_attn_dlr100"),
    ],
    "small / FineWeb-Edu": [
        ("Attention only, ×30", "small_fineweb_dendrites_attn_dlr30"),
        ("Attention only, ×100, 6 dendrites", "small_fineweb_dendrites_attn_dlr100_d6"),
        ("Attention only in top 2 layers (2-3), ×30", "small_fineweb_dendrites_attn_top2_dlr30"),
        ("Wide in top layer (3), ×30, 6 dendrites", "small_fineweb_dendrites_wide_top1_dlr30_d6"),
        ("Wide in top layer (3), ×100, 6 dendrites", "small_fineweb_dendrites_wide_top1_dlr100_d6"),
        ("Attention only in top layer (3), ×30, 6 dendrites", "small_fineweb_dendrites_attn_top1_dlr30_d6"),
    ],
    "mini / FineWeb-Edu, seed 7 (different data order, dendrite init and eval masks)": [
        ("Wide in top layer (3), ×30, 6 dendrites", "mini_fineweb_dendrites_wide_top1_dlr30_d6_seed7"),
    ],
    "mini / FineWeb-Edu, 2M training samples per epoch (4× longer cycles, same baseline and eval set)": [
        ("Wide in top layer (3), ×30, 6 dendrites", "mini_fineweb2m_dendrites_wide_top1_dlr30_d6"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×1", "mini_fineweb2m_ffn_control_top1_dlr1_d6"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×3", "mini_fineweb2m_ffn_control_top1_dlr3_d6"),
    ],
    "small / FineWeb-Edu, 2M training samples per epoch (4× longer cycles, same baseline and eval set)": [
        ("Wide in top layer (3), ×30, 6 dendrites", "small_fineweb2m_dendrites_wide_top1_dlr30_d6"),
    ],
    "mini / CNN/DailyMail": [
        ("Wide in top layer (3), ×30, 6 dendrites", "mini_cnn_dendrites_wide_top1_dlr30_d6"),
        ("No dendrites: 10199 extra FFN neurons in layer 3 (as many parameters as 6 top-layer dendrites), ×1", "mini_cnn_ffn_control_top1_dlr1_d6"),
    ],
    "mini / FineWeb-Edu, from the pretrained checkpoint (no stage 1; the control is stage 1 itself)": [
        ("Wide, ×100, 6 dendrites", "mini_fineweb_dendrites_from_pretrained_wide_dlr100"),
        ("Attention only in top 2 layers (2-3), ×30, 6 dendrites", "mini_fineweb_dendrites_from_pretrained_attn_top2_dlr30"),
    ],
}


def control_at(control, cycle):
    """Best control score up to `cycle`; a control that stopped improving keeps its best score."""
    return min(score for c, score in control.items() if c <= cycle)


lines = [
    "# PerforatedAI dendrites on BERT MLM: experiment log",
    "",
    f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from the `*.log` files in `{ROOT}`.",
    "",
    "Every run starts from a converged stage-1 baseline (LR restarts until a cycle gains < 0.1%). A cycle is one",
    "3-epoch LR schedule (2e-5, linear decay), and a dendrite run adds one dendrite per cycle. Each cell is the best",
    "eval loss of that cycle (fixed eval masks), and Δ is the difference against the restart-only control at the same",
    "cycle, which got the same restarts without dendrites. ×N is `--pai_dendrite_lr_multiplier`. Placements: *wide* is",
    "every encoder Linear plus the MLM head dense, *attention* is q/k/v/attention-output plus head, *FFN* is",
    "intermediate/output dense plus head.",
    "",
]
for section, control_names in CONTROLS.items():
    control_names = [n for n in ((control_names,) if isinstance(control_names, str) else control_names) if n in runs]
    if not control_names:
        continue
    control_name = max(control_names, key=lambda n: len(runs[n]["cycles"]))
    control = runs[control_name]["cycles"]
    present = [n for _, name in EXPERIMENTS[section] for n in (name, f"{name}_INTERRUPTED") if n in runs]
    if not present:
        continue
    max_cycle = max(max(runs[name]["cycles"]) for name in present)
    header = "| Run | Params after last cycle | " + " | ".join(f"Cycle {c}" for c in range(1, max_cycle + 1)) + " |"
    lines += [f"## {section}", "", header, "|" + "---|" * (max_cycle + 2)]
    lines.append(
        "| **Control (restarts only)** | "
        + f"{runs[control_name]['params'][0] / 1e6:.1f}M | "
        + " | ".join(f"{control_at(control, c):.4f}" for c in range(1, max_cycle + 1))
        + " |"
    )
    for label, name in EXPERIMENTS[section]:
        if name not in runs and f"{name}_INTERRUPTED" in runs:
            # Runs stopped for the 2026-10-11 reboot keep their completed cycles until they are rerun.
            name, label = f"{name}_INTERRUPTED", f"{label} (interrupted)"
        if name not in runs:
            continue
        run = runs[name]
        cells = []
        for c in range(1, max_cycle + 1):
            if c in run["cycles"]:
                delta = run["cycles"][c] - control_at(control, c)
                cells.append(f"{run['cycles'][c]:.4f} ({delta:+.4f})")
            else:
                cells.append("")
        params = run["params"][min(len(run["params"]), max(run["cycles"]) + 1) - 1] if run["params"] else 0
        lines.append(f"| {label} | {params / 1e6:.1f}M | " + " | ".join(cells) + " |")
    lines.append("")

open(f"{ROOT}/pai_bert_dendrite_experiments.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
