#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team All rights reserved.
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

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import datetime
import logging
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import chain

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

BERT_SHORTCUT_NAMES = {
    "tiny": "google/bert_uncased_L-2_H-128_A-2",
    "mini": "google/bert_uncased_L-4_H-256_A-4",
    "small": "google/bert_uncased_L-4_H-512_A-8",
    "medium": "google/bert_uncased_L-8_H-512_A-8",
    "base": "bert-base-uncased",
    "large": "bert-large-uncased",
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: str | None = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: str | None = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: str | None = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.model_name_or_path in BERT_SHORTCUT_NAMES:
            self.model_shortcut = self.model_name_or_path
            self.model_name_or_path = BERT_SHORTCUT_NAMES[self.model_name_or_path]
        else:
            self.model_shortcut = None
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str | None = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: str | None = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: str | None = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: str | None = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: int | None = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: int | None = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    text_column_name: str | None = field(
        default=None,
        metadata={"help": "Name of the text column in the dataset. Defaults to 'text' if present, otherwise the first column."},
    )
    fixed_eval_masks: bool = field(
        default=True,
        metadata={
            "help": (
                "Mask the evaluation set once using `--seed` instead of re-masking it on every evaluation, so "
                "unchanged weights always get the same eval loss. Not applied with `--streaming`."
            )
        },
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")


@dataclass
class PerforatedAIArguments:
    """
    Arguments for the two-stage PerforatedAI protocol: first train a baseline with LR restarts only until restarts stop
    helping, then add dendrites on top of that saved baseline so that any further gain can be credited to dendrites.
    """

    pai_stage: str = field(
        default="dendrites",
        metadata={
            "help": (
                "`baseline`: restart the LR schedule without adding dendrites until restarts stop improving, then save "
                "the best model to `--output_dir` so it can be passed as `--model_name_or_path` to a `dendrites` run. "
                "`dendrites`: add dendrites to `--model_name_or_path`."
            ),
            "choices": ["baseline", "dendrites"],
        },
    )
    pai_switch_on_schedule_end: bool = field(
        default=True,
        metadata={
            "help": (
                "Restart the LR schedule (adding dendrites in the `dendrites` stage) every `--num_train_epochs` epochs, "
                "when the schedule has decayed to its end, instead of waiting for PAI's `n_epochs_to_switch` epochs "
                "without improvement while the LR sits at 0."
            )
        },
    )
    pai_improvement_threshold: float = field(
        default=0.001,
        metadata={
            "help": (
                "Relative eval loss improvement a full cycle (one LR schedule, plus a new dendrite in the `dendrites` "
                "stage) needs over the best previous cycle. Training ends at the first cycle that falls short, so the "
                "same bar decides when the baseline has plateaued and when to stop adding dendrites."
            )
        },
    )
    pai_max_dendrites: int | None = field(
        default=None,
        metadata={
            "help": (
                "Maximum number of dendrites to add in the `dendrites` stage (default 6), or maximum number of LR "
                "restarts in the `baseline` stage (default 100, so the plateau check decides when to stop)."
            )
        },
    )
    pai_dendrite_top_layers: int = field(
        default=1,
        metadata={
            "help": (
                "Add dendrites to the Linear layers of this many top encoder layers and to the MLM head's transform "
                "dense; all other layers are left unchanged. Dendrites closest to the output give the most gain per "
                "added parameter. Ignored when `--pai_config_file` is given."
            )
        },
    )
    pai_config_file: str | None = field(
        default=None,
        metadata={
            "help": (
                "PAI config JSON with custom perforation targets for the `dendrites` stage, instead of "
                "`--pai_dendrite_top_layers`."
            )
        },
    )
    pai_dendrite_lr_multiplier: float = field(
        default=30.0,
        metadata={
            "help": (
                "Multiplies the learning rate of dendrite parameters. Their output weights start at 0, so at a "
                "fine-tuning learning rate they barely move within a cycle and the dendrites add nothing."
            )
        },
    )


class DataCollatorForMLMWithFixedEvalMasks(DataCollatorForLanguageModeling):
    """
    Masks batches on the fly like [`DataCollatorForLanguageModeling`], except for examples that were already masked
    ahead of time and carry their own `labels`, which are only padded.
    """

    def torch_call(self, examples):
        if not (isinstance(examples[0], Mapping) and "labels" in examples[0]):
            return super().torch_call(examples)
        labels = [example["labels"] for example in examples]
        features = [{key: value for key, value in example.items() if key != "labels"} for example in examples]
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, features, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )
        batch["labels"] = torch.full(batch["input_ids"].shape, -100, dtype=torch.long)
        for i, example_labels in enumerate(labels):
            if self.tokenizer.padding_side == "left":
                batch["labels"][i, -len(example_labels) :] = torch.tensor(example_labels)
            else:
                batch["labels"][i, : len(example_labels)] = torch.tensor(example_labels)
        return batch


# Parameters PAI adds for dendrites. `dendrite_module.parent_module` is a stale copy of the wrapped module, not a dendrite.
DENDRITE_PARAMETER_MARKERS = (
    ".dendrite_module.layers.",
    ".dendrites_to_top.",
    ".dendrite_module.dendrites_to_dendrites.",
)


class PAITrainer(Trainer):
    """[`Trainer`] that can train dendrite parameters with a larger learning rate than the rest of the model."""

    def __init__(self, *args, dendrite_lr_multiplier=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dendrite_lr_multiplier = dendrite_lr_multiplier

    def create_optimizer(self, model=None):
        if self.optimizer is not None or self.dendrite_lr_multiplier == 1.0:
            return super().create_optimizer(model)
        opt_model = self.model if model is None else model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        groups = {}
        for name, param in opt_model.named_parameters():
            if param.requires_grad:
                is_main = not any(marker in name for marker in DENDRITE_PARAMETER_MARKERS)
                groups.setdefault((is_main, name in decay_parameters), []).append(param)
        # Dendrite groups go first because PAI records the learning rate of the last parameter group.
        self.optimizer = optimizer_cls(
            [
                {
                    "params": params,
                    "weight_decay": self.args.weight_decay if decays else 0.0,
                    "lr": optimizer_kwargs["lr"] * (1.0 if is_main else self.dendrite_lr_multiplier),
                }
                for (is_main, decays), params in sorted(groups.items())
            ],
            **optimizer_kwargs,
        )
        return self.optimizer


class PAICyclePlateauCallback(TrainerCallback):
    """
    Ends PAI training at the first cycle whose best eval loss beats the best of all previous cycles by less than
    `min_improvement` (relative), by capping `max_dendrites` right before PAI handles the end of that cycle.

    PAI's own improvement threshold is also applied between the epochs of a cycle, so a cycle whose last epoch only
    slightly beats its earlier epochs would be rejected even if it clearly beat the previous best. PAI's threshold is
    therefore set to 0 and the plateau is checked here on whole cycles instead. Requires `DOING_FIXED_SWITCH`.
    """

    def __init__(self, min_improvement):
        self.min_improvement = min_improvement
        self.best_before_cycle = None
        self.cycle_best = None
        self.training_ended = False

    def on_train_end(self, args, state, control, **kwargs):
        # Evaluations after training are not part of any cycle.
        self.training_ended = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.training_ended or metrics is None or "eval_loss" not in metrics:
            return
        score = metrics["eval_loss"]
        self.cycle_best = score if self.cycle_best is None else min(self.cycle_best, score)
        # Same check PAI's switch_time() does for DOING_FIXED_SWITCH when this score is added right after this callback.
        tracker_vars = GPA.pai_tracker.member_vars
        last_switch = tracker_vars["switch_epochs"][-1] if tracker_vars["switch_epochs"] else 0
        if tracker_vars["num_epochs_run"] - last_switch < GPA.pc.get_n_fixed_switch_num():
            return
        if self.best_before_cycle is not None:
            improvement = (self.best_before_cycle - self.cycle_best) / self.best_before_cycle
            logger.info(
                f"PAI cycle {tracker_vars['num_dendrites_added']} ended with best eval_loss {self.cycle_best:.6f}, "
                f"{improvement:.4%} better than the previous best {self.best_before_cycle:.6f}"
            )
            if improvement < self.min_improvement:
                logger.info(f"Improvement is below {self.min_improvement:.4%}, ending PAI training after this cycle")
                # PAI finishes and reloads its best model instead of restarting once this many dendrites are added.
                GPA.pc.set_max_dendrites(tracker_vars["num_dendrites_added"])
        else:
            logger.info(f"PAI cycle 0 ended with best eval_loss {self.cycle_best:.6f}")
        if self.best_before_cycle is None or self.cycle_best < self.best_before_cycle:
            self.best_before_cycle = self.cycle_best
        self.cycle_best = None


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PerforatedAIArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, pai_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, pai_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=dtype,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # PerforatedAI configuration
    doing_pai = pai_args.pai_stage == "dendrites"
    GPA.pc.set_library_validation_score("eval_loss")
    GPA.pc.set_library_extra_scores(["train_loss"])
    GPA.pc.set_library_extra_scores_without_graphing(["eval_perplexity", "eval_accuracy"])
    # With doing_pai=False each "dendrite" is an LR restart, so this also caps the number of baseline restarts.
    if pai_args.pai_max_dendrites is not None:
        GPA.pc.set_max_dendrites(pai_args.pai_max_dendrites)
    else:
        GPA.pc.set_max_dendrites(6 if doing_pai else 100)
    if pai_args.pai_switch_on_schedule_end:
        if training_args.do_train and (
            training_args.eval_strategy != "epoch" or not float(training_args.num_train_epochs).is_integer()
        ):
            raise ValueError(
                "--pai_switch_on_schedule_end requires --eval_strategy epoch and an integer --num_train_epochs"
            )
        # The trainer builds every LR schedule to last num_train_epochs, so switch as soon as it reaches the end.
        GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_n_fixed_switch_num(int(training_args.num_train_epochs))
        # Let PAI keep every improvement and apply the threshold to whole cycles in PAICyclePlateauCallback.
        GPA.pc.set_improvement_threshold([0.0])
    else:
        # PAI indexes a list threshold by the number of cycles so far and the default decays to 0, which would let
        # tiny gains keep the baseline restarting forever, so use a single value.
        GPA.pc.set_improvement_threshold([pai_args.pai_improvement_threshold])
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_module_names_to_perforate(["Linear"])
    GPA.pc.set_output_dimensions([-1, -1, 0])  # [batch, sequence, features]
    if not doing_pai:
        # No dendrites are added, so the perforation targets don't matter and the interactive selection is skipped.
        GPA.pc.set_configuration_confirmed(True)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        # Don't retry a restart that didn't improve: without dendrites PAI's retry path records a single switch epoch,
        # which breaks its best_arch_scores bookkeeping. The first cycle of the `dendrites` stage is another restart
        # from the same weights without dendrites, so it double-checks the plateau.
        GPA.pc.set_max_dendrite_tries(1)
    elif pai_args.pai_config_file is None:
        if not hasattr(model, "bert"):
            raise ValueError("--pai_dendrite_top_layers only supports BERT models, pass --pai_config_file instead")
        # PAI adds dendrites to every Linear it doesn't track. The decoder is tied to the word embeddings and a
        # dendrite copy of it would add vocab_size x hidden_size parameters.
        num_layers_without_dendrites = max(config.num_hidden_layers - pai_args.pai_dendrite_top_layers, 0)
        GPA.pc.set_module_names_to_track(["BertEmbeddings", "LayerNorm"])
        GPA.pc.set_module_ids_to_track(
            [".cls.predictions.decoder"] + [f".bert.encoder.layer.{i}" for i in range(num_layers_without_dendrites)]
        )
        GPA.pc.set_parameter_ids_to_track([".cls.predictions.bias"])
        GPA.pc.set_configuration_confirmed(True)
    if model_args.model_shortcut:
        _model_label = model_args.model_shortcut
    elif os.path.isdir(model_args.model_name_or_path):
        _model_label = os.path.basename(os.path.normpath(model_args.model_name_or_path))
    else:
        _model_label = model_args.model_name_or_path.replace('/', '_')
    _pai_config_name = (
        f"{_model_label}"
        f"_{(data_args.dataset_name or 'custom').replace('/', '_')}"
    )
    # The process id keeps runs launched in the same second from sharing a folder and loading each other's models.
    _pai_save_name = (
        f"{_pai_config_name}_{pai_args.pai_stage}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    )
    model = UPA.perforate_model(
        model,
        save_name=_pai_save_name,
        maximizing_score=False,
        making_graphs=True,
        config_file=pai_args.pai_config_file if doing_pai else None,
        doing_pai=doing_pai,
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "text" in column_names:
        text_column_name = "text"
    else:
        text_column_name = column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=[text_column_name],
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
            if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )
            else:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with training_args.main_process_first(desc="grouping texts together"):
            if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)

    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForMLMWithFixedEvalMasks(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    if training_args.do_eval and data_args.fixed_eval_masks and not data_args.streaming:
        # Re-masking on every evaluation makes the eval loss of unchanged weights jump around by more than the gains
        # PAI is trying to detect, so draw the masks once. The same seed gives both PAI stages the same eval masks.
        eval_masker = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
        eval_masker.generator = torch.Generator().manual_seed(training_args.seed)

        def mask_eval_examples(examples):
            masked_input_ids, labels = [], []
            for input_ids, special_tokens_mask in zip(examples["input_ids"], examples["special_tokens_mask"]):
                example_input_ids, example_labels = eval_masker.torch_mask_tokens(
                    torch.tensor([input_ids]), special_tokens_mask=torch.tensor([special_tokens_mask])
                )
                masked_input_ids.append(example_input_ids[0].tolist())
                labels.append(example_labels[0].tolist())
            return {"input_ids": masked_input_ids, "labels": labels}

        eval_dataset = eval_dataset.map(
            mask_eval_examples,
            batched=True,
            remove_columns=["special_tokens_mask"],
            load_from_cache_file=False,
            keep_in_memory=True,
            desc="Masking evaluation set once",
        )

    class PAIMetricsCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            if "eval_loss" in metrics:
                try:
                    metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
                except OverflowError:
                    metrics["eval_perplexity"] = float("inf")

    # Initialize our Trainer
    trainer = PAITrainer(
        dendrite_lr_multiplier=pai_args.pai_dendrite_lr_multiplier,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
        perforate_model=True,
        callbacks=[PAIMetricsCallback()]
        + ([PAICyclePlateauCallback(pai_args.pai_improvement_threshold)] if pai_args.pai_switch_on_schedule_end else []),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        if not doing_pai and trainer.is_world_process_zero():
            # trainer.save_model() is a no-op while perforating, and PAI's own saves keep its module wrappers. Save
            # the best baseline as a plain checkpoint so a `dendrites` run can load it with --model_name_or_path.
            # PAI wraps converted modules, which inserts `.main_module` into their parameter names.
            pai_state_dict = {
                key.replace(".main_module", ""): value
                for key, value in trainer.accelerator.unwrap_model(trainer.model).state_dict().items()
            }
            baseline_model = AutoModelForMaskedLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
            baseline_model.load_state_dict(
                {key: pai_state_dict[key] for key in baseline_model.state_dict()}, strict=True
            )
            baseline_model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            logger.info(f"Saved PAI baseline to {training_args.output_dir}")
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
