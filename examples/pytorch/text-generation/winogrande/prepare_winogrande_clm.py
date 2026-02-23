#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _format_example(example, include_answer=True):
    sentence = example["sentence"].strip()
    option1 = example["option1"].strip()
    option2 = example["option2"].strip()
    label = example.get("label", None)

    prompt = (
        f"Sentence: {sentence}\n"
        f"Option1: {option1}\n"
        f"Option2: {option2}\n"
        "Answer:"
    )

    answer = ""
    if label is not None and label != -1:
        if str(label) == "1":
            answer = option1
        elif str(label) == "2":
            answer = option2

    if include_answer and answer:
        text = f"{prompt} {answer}"
    else:
        text = prompt

    return {
        "text": text,
        "prompt": prompt,
        "answer": answer,
        "label": str(label) if label is not None else None,
    }


def _write_jsonl(dataset, path, include_answer=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in dataset:
            record = _format_example(example, include_answer=include_answer)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Winogrande for CLM fine-tuning")
    parser.add_argument("--dataset_name", type=str, default="winogrande")
    parser.add_argument("--dataset_config_name", type=str, default="winogrande_xl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--include_answer_in_validation",
        action="store_true",
        help="Include the answer in validation text (default: False).",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, args.dataset_config_name)

    output_dir = Path(args.output_dir)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"

    _write_jsonl(dataset["train"], train_path, include_answer=True)

    include_answer = args.include_answer_in_validation
    _write_jsonl(dataset["validation"], validation_path, include_answer=include_answer)


if __name__ == "__main__":
    main()
