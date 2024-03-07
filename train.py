import os
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, get_kbit_device_map
import torch
import argparse

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def load_message_dataset(dataset_name, eval_data_size, dataset_col_name):
    dataset = load_dataset(dataset_name)
    dataset = dataset.rename_column(dataset_col_name, "messages")
    dataset = dataset.select_columns(["messages"])

    if "test" not in dataset.keys() and eval_data_size > 0:
        dataset = dataset["train"].train_test_split(eval_data_size)

    return dataset

def load_unsloth_model(args):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.quant_size == 4,
    )

    loftq_config = {} if args.do_loftq else None

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = args.lora_alpha,
        lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = args.do_gradient_checkpointing,
        random_state = args.random_state,
        use_rslora = args.do_rslora,  # We support rank stabilized LoRA
        loftq_config = loftq_config, # And LoftQ
    )

    peft_config = None

    return model, tokenizer, peft_config


def load_hf_model(args):

    if args.quant_size == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=None,
            bnb_4bit_quant_type="nf4", # nf4 or fp4
            bnb_4bit_use_double_quant=True,
        )

    elif args.quant_size == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    model_kwargs = dict(
        trust_remote_code=args.trust_remote_code,
        torch_dtype=None,
        use_cache=False if args.do_gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    if args.do_lora:
        loftq_config = {} if args.do_loftq else None

        peft_config = LoraConfig(
            target_modules="all-linear",
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=args.do_dora,
            loftq_config=loftq_config,
            use_rslora=args.do_rslora
        )
    else:
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model_kwargs, tokenizer, peft_config

def get_trainer(model_name, model, tokenizer, dataset, peft_config, args):

    # run_name = model_name + "__" + dataset_name
    run_name = args.run_name.replace("/", "__")

    training_args = TrainingArguments(
        num_train_epochs = args.num_epochs,
        report_to="all",
        run_name=run_name,
        per_device_train_batch_size = args.train_batch_size,
        per_device_eval_batch_size = args.eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        warmup_steps = args.warmup_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate = args.learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        neftune_noise_alpha=args.neftune_noise_alpha,
        optim = args.optimizer,
        weight_decay = args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        seed = args.random_state,
        output_dir = args.output_folder,
    )

    trainer = SFTTrainer(
        model = model if args.do_unsloth else model_name,
        model_init_kwargs = None if args.do_unsloth else model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        max_seq_length = args.max_seq_length,
        packing = args.do_packing, # Can make training 5x faster for short sequences.
        peft_config = peft_config,
        args = training_args
    )

    return trainer

def run_training(args):
    dataset = load_message_dataset(args.dataset_name, args.eval_data_size, args.dataset_col_name)

    if args.do_unsloth:
        model, tokenizer, peft_config = load_unsloth_model(args)
    else:
        model, tokenizer, peft_config = load_hf_model(args)

    trainer = get_trainer(args.model_name, model, tokenizer, dataset, peft_config, args)

    trainer.evaluate()
    trainer.train()
    trainer.save_model(args.output_folder)

# Parses the arguments and runs the training
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="lightblue/multi_context_closed_qa", help="The name of the dataset to use")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="The name of the model to use")
    parser.add_argument("--dataset_col_name", type=str, default="closedqa_messages", help="The name of the column in the dataset to use")
    parser.add_argument("--eval_data_size", type=int, default=0.001, help="The maximum sequence length")
    parser.add_argument("--max_seq_length", type=int, default=32000, help="The maximum sequence length")
    parser.add_argument("--quant_size", type=int, default=4, help="The quantization size")
    parser.add_argument("--do_loftq", type=bool, default=False, help="Whether to use LoFTQ")
    parser.add_argument("--lora_rank", type=int, default=32, help="The rank of the LoRA matrix")
    parser.add_argument("--lora_alpha", type=float, default=16, help="The alpha parameter of LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="The dropout rate of LoRA")
    parser.add_argument("--do_gradient_checkpointing", type=bool, default=False, help="Whether to use gradient checkpointing")
    parser.add_argument("--random_state", type=int, default=123, help="The random state")
    parser.add_argument("--do_rslora", type=bool, default=False, help="Whether to use rank-stabilized LoRA")
    parser.add_argument("--loftq_config", type=dict, default={}, help="The LoFTQ config")
    parser.add_argument("--do_lora", type=bool, default=False, help="Whether to use LoRA")
    parser.add_argument("--do_dora", type=bool, default=False, help="Whether to use DoRA")
    parser.add_argument("--trust_remote_code", type=bool, default=False, help="trust_remote_code model?")
    parser.add_argument("--num_epochs", type=int, default=3, help="The number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=1, help="The training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="The evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=5, help="The warmup steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit", help="The optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="The learning rate scheduler type")
    parser.add_argument("--output_folder", type=str, default="trl_trained_model", help="The output folder")
    parser.add_argument("--do_packing", type=bool, default=False, help="Whether to use packing")
    parser.add_argument("--neftune_noise_alpha", type=float, default=0.0, help="The Neftune noise alpha")
    parser.add_argument("--run_name", type=str, default="run", help="The run name")
    parser.add_argument("--do_unsloth", type=bool, default=False, help="Whether to use UnSLoth")
    args = parser.parse_args()

    run_training(args)

if __name__ == "__main__":
    main()