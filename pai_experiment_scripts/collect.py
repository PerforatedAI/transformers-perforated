"""Collect per-cycle best eval loss and parameter counts for every PAI experiment log in the repo root."""
import csv
import glob
import os
import re

ROOT = "/home/rbrenner/PAI/PerforatedAI/transformers-perforated"


def cycles(log_path):
    """Return {cycle: best eval_loss} from PAICyclePlateauCallback log lines."""
    out = {}
    for line in open(log_path, errors="ignore"):
        m = re.search(r"PAI cycle (\d+) ended with best eval_loss ([\d.]+)", line)
        if m:
            out.setdefault(int(m.group(1)), float(m.group(2)))
    return out


def pai_folder(log_path):
    for line in open(log_path, errors="ignore"):
        m = re.search(r"Saved \d+ variables → (\S+)_config\.json", line)
        if m:
            return os.path.dirname(m.group(1) + "_config.json")
    return None


def param_counts(folder):
    files = glob.glob(os.path.join(folder, "*_best_arch_scores.csv")) if folder else []
    files = [f for f in files if "beforeSwitch" not in f and "before_final" not in f and "noImprove" not in f]
    if not files:
        return []
    rows = list(csv.reader(open(files[0])))[1:]
    return [int(float(r[0])) for r in rows if r]


runs = {}
for log in sorted(glob.glob(os.path.join(ROOT, "*.log"))):
    name = os.path.basename(log)[:-4]
    c = cycles(log)
    if c:
        runs[name] = {"cycles": c, "params": param_counts(pai_folder(log)), "folder": pai_folder(log)}

if __name__ == "__main__":
    for name, r in runs.items():
        print(name, r["cycles"], r["params"][:8], os.path.basename(r["folder"] or ""))
