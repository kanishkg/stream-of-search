import argparse
import json
import os
import random

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import GPTNeoConfig
from transformers import GPTNeoForCausalLM
from transformers import Trainer
from transformers import TrainingArguments

import wandb


def main(args):
    # read config from a json config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # set seeds
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up accelerator
    accelerator = Accelerator()

    if args.wandb and accelerator.is_main_process:
        wandb_kwargs = config.get(
            "wandb",
            {
                "project": "",
                "entity": "",
                "dir": "",
            },
        )
        wandb.init(
            project=wandb_kwargs["project"],
            entity=wandb_kwargs["entity"],
            name=config["name"],
            config=config,
            dir=wandb_kwargs["dir"],
        )

    with open(config["model_config"], "r") as f:
        model_config = json.load(f)

    # only GPTNeo model for now
    if not args.reset:
        model_config = GPTNeoConfig(**model_config)
        model = GPTNeoForCausalLM(model_config)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    else:
        model = GPTNeoForCausalLM.from_pretrained(
            args.ckpt,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

    # print(f"Number of parameters: {model.num_parameters()}")

    # load dataset
    train_file = os.path.join(config["data_dir"], config["train_file"])
    val_file = os.path.join(config["data_dir"], config["val_file"])
    val_target_file = os.path.join(config["data_dir"], config["val_target_file"])
    hf_datasets = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "val": val_file,
            "val_target": val_target_file,
        },
    )
    hf_datasets["train"] = hf_datasets["train"].select(range(int(config["num_train"])))

    context_length = config["context_length"]
    tokenizer.model_max_length = context_length

    def tokenize(element):
        if config["train_type"] == "dt":
            text = [
                tokenizer.bos_token
                + f"{element['rating'][e]:0.2f}->"
                + element["search_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["search_path"]))
            ]
        elif config["train_type"] == "sft":
            text = [
                tokenizer.bos_token
                + element["search_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["search_path"]))
            ]
        elif config["train_type"] == "oft":
            text = [
                tokenizer.bos_token
                + element["optimal_path"][e].strip()
                + tokenizer.eos_token
                for e in range(len(element["optimal_path"]))
            ]
        else:
            raise ValueError(f"Invalid train type: {config['train_type']}")
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
            stride=0,
            padding="max_length",
        )
        return {"input_ids": outputs["input_ids"]}

    # tokenize dataset for causal LM
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_datasets = hf_datasets.map(
        tokenize, batched=True, remove_columns=hf_datasets["train"].column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print("tokenized dataset", tokenized_datasets)

    # prepare training
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        logging_steps=config["log_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=True,
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        learning_rate=config["lr"],
        save_strategy="steps",
        save_total_limit=config["save_total_limit"],
        save_steps=config["save_steps"],
        seed=config["seed"],
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        run_name=config["name"],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        torch_compile=True,
        metric_for_best_model="valid_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset={
            "valid": tokenized_datasets["val"],
            "valid_target": tokenized_datasets["val_target"],
        },
    )

    # train
    if args.resume:
        trainer.train(resume_from_checkpoint=args.ckpt)
    else:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../../configs/conf.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    main(args)
