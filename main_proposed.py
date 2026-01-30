from utils import client_selection, GeneralClient
from tqdm import tqdm
import fire
import os, json, random
import torch
import copy
import numpy as np
from typing import List
from datetime import datetime
from torch.nn import CrossEntropyLoss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from transformers import (
    AutoTokenizer, LlamaForCausalLM,LlamaTokenizer,
    AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, TrainerCallback
)
from peft import (
    LoraConfig, prepare_model_for_kbit_training,
    get_peft_model, PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from collections import defaultdict
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_text(path, text):
    f = open(path, "a", encoding="utf-8")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(text + "\n")

def format_example_json(example, include_answer=True):
    prompt = example["instruction"].rstrip()
    prompt += "\nAnswer: "
    if include_answer:
        label = str(example["response"]).strip()
        prompt += f"{label}\n\n"
    return prompt

def validate(model, tokenizer, valid_data, max_length=2048):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    losses = []
    with torch.no_grad():
        for ex in valid_data:
            full_text = format_example_json(ex, include_answer=True)
            tokenized = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(device)
            outputs = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask"),
                labels=tokenized["input_ids"]
            )
            loss = outputs.loss
            losses.append(loss.item())
    return float(np.mean(losses))

def task_validate(model, tokenizer, valid_data, max_length=2048):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    task_losses = defaultdict(list)

    with torch.no_grad():
        for ex in valid_data:
            task = ex.get("category", "unknown")
            full_text = format_example_json(ex, include_answer=True)
            tokenized = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(device)
            outputs = model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized.get("attention_mask"),
                labels=tokenized["input_ids"]
            )
            loss = outputs.loss.item()
            task_losses[task].append(loss)

    task_avg_losses = {
        task: float(np.mean(losses))
        for task, losses in task_losses.items()
    }

    overall_loss = float(
        np.mean([l for losses in task_losses.values() for l in losses])
    )

    return overall_loss, task_avg_losses

def load_lora_state(lora_dir):
    return torch.load(
        os.path.join(lora_dir, "pytorch_model.bin"),
        map_location="cpu"
    )

def lora_state_to_vector(state_dict):
    return torch.cat([v.flatten() for v in state_dict.values()])

def vector_to_lora_state(vec, ref_state):
    new_sd = {}
    idx = 0
    for k, v in ref_state.items():
        numel = v.numel()
        new_sd[k] = vec[idx:idx+numel].reshape(v.shape)
        idx += numel
    return new_sd

def balanced_model_merging(
        global_model: str = '',
        data_path: str = './data',
        output_dir: str = './lora-model_/',
        shot:int=0,
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 10,
        num_clients: int = 5,
        local_batch_size: int = 8,
        local_micro_batch_size: int = 2,
        local_num_epochs: int = 1,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        lora_target_modules: List[str] = [
            "q_proj","k_proj","v_proj","o_proj"
        ],
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,
        round_type: str = "r-1",
        scaling_coef_list: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        temp_list: List[float] = [2,2,2,2,2],
        lamda_value: float = 0.3,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"shot: {shot}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    set_seed(1234)

    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each task")
    global_valid_path = os.path.join(data_path, "global_valid.json")
    from datasets import load_dataset
    global_valid = load_dataset(
        "json",
        data_files=global_valid_path,
        streaming=False
    )["train"]

    gradient_accumulation_steps = local_batch_size // local_micro_batch_size

    if global_model == "google/gemma-2-2b" or "Qwen/Qwen2.5-0.5B":
        model = AutoModelForCausalLM.from_pretrained(global_model, torch_dtype=torch.float16)
        model.config.attn_implementation = "eager"
    else:
        model = LlamaForCausalLM.from_pretrained(
            global_model,
            torch_dtype=torch.float16,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        global_model,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    output_dir = os.path.join(output_dir, str(num_clients))

    def preprocess_wrapper(ex, all_examples, round=None, client_id=None, output_dir=None):
        set_seed(1234)
        full_text = format_example_json(ex, include_answer=True)
        os.makedirs(output_dir, exist_ok=True)
        debug_file = os.path.join(output_dir, f"round{round}_client{client_id}.txt")
        save_text(debug_file, "=====================================")
        save_text(debug_file, full_text + "\n")
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=cutoff_len,
            padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    try:
        model.config.use_cache = False
    except:
        pass
    try:
        model.config.attn_implementation = "sdpa"
    except:
        pass

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    previously_selected_clients_set = set()
    local_dataset_len_dict = dict()

    prev_global_lora = {k: v.detach().cpu().clone()
                        for k, v in get_peft_model_state_dict(model, adapter_name="default").items()}

    for round in tqdm(range(num_communication_rounds)):
        selected_clients_set = [0, 1, 2, 3]

        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir)

            client.preprare_local_dataset(preprocess_wrapper, round)
            client.local_val_set_size = local_val_set_size

            client.build_local_trainer(
            tokenizer,
            local_micro_batch_size,
            gradient_accumulation_steps,
            local_num_epochs,
            local_learning_rate,
            group_by_length,
            )

            print("Initiating the local training of task_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of task_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = (
            client.terminate_local_training(
                round,
                local_dataset_len_dict,
                previously_selected_clients_set
                )
            )
            del client

        client_lora = {}
        for client_id in selected_clients_set:
            path = os.path.join(
                output_dir,
                str(round),
                f"local_output_{client_id}",
                "pytorch_model.bin"
            )
            client_lora[client_id] = torch.load(path, map_location="cpu")

        print("Collecting the weights of clients and performing aggregation")

        if round_type == "r-1":
            client_prev_global_loss = {}
            client_local_loss = {}
            gaps = []
            selected_clients = list(selected_clients_set)
            TEMP_SCHEDULE = temp_list

            CLIENT_TASK_MAP = {
                0: "instruction",
                1: "safety",
                2: "multilingual",
                3: "math",
            }

            for cid in selected_clients:
                task = CLIENT_TASK_MAP[cid]
                client_valid = [ex for ex in global_valid if ex.get("category") == task]

                backup_lora = get_peft_model_state_dict(model, "default")
                set_peft_model_state_dict(model, prev_global_lora, "default")
                L_MTL = validate(model, tokenizer, client_valid)

                set_peft_model_state_dict(model, client_lora[cid], "default")
                L_local = validate(model, tokenizer, client_valid)

                set_peft_model_state_dict(model, backup_lora, "default")

                client_prev_global_loss[cid] = float(L_MTL)
                client_local_loss[cid] = float(L_local)

                g_i = float(L_MTL - L_local)
                gaps.append(g_i)

        gaps = np.array(gaps, dtype=np.float64)
        w = np.exp(gaps / TEMP_SCHEDULE[round])
        alphas = w / (w.sum() + 1e-12)

        lora_dirs = [
            os.path.join(output_dir, str(round), f"local_output_{cid}")
            for cid in selected_clients_set
        ]
        lora_states = [load_lora_state(d) for d in lora_dirs]
        task_vecs = [lora_state_to_vector(sd) for sd in lora_states]

        k = 2

        weighted_task_vecs = [a * tv for a, tv in zip(alphas, task_vecs)]
        mtl_tv = torch.stack(weighted_task_vecs, dim=0).sum(dim=0)
        tall_masks = []
        for alpha, task_vec in  zip(alphas, task_vecs):
            tall_mask = torch.abs(alpha*task_vec) > torch.abs(mtl_tv - alpha*task_vec) * lamda_value
            tall_masks.append(tall_mask)
        consensus_mask = torch.zeros_like(tall_masks[0], dtype=torch.int16)
        for mask in tall_masks:
            consensus_mask += mask.to(torch.int16)

        consensus_mask = consensus_mask >= k
        merged_tv = (mtl_tv * consensus_mask * scaling_coef_list[round])
        merged_state = vector_to_lora_state(
            merged_tv,
            ref_state=lora_states[0]
        )

        set_peft_model_state_dict(model, merged_state, "default")

        adapter_path = os.path.join(output_dir, str(round), f"adapter_model_round{round}.bin")
        torch.save(merged_state, adapter_path)
        adapter_path = os.path.join(output_dir, f"adapter_model.bin")
        torch.save(merged_state, adapter_path)
        print(f"[Round {round}] Saved global LoRA adapter → {adapter_path}")

        overall_loss, task_losses = task_validate(
           model, tokenizer, global_valid
        )

        prev_global_lora = {k: v.detach().cpu().clone() for k, v in merged_state.items()}

if __name__ == "__main__":
   fire.Fire({
       "run_balanced_model_merging": balanced_model_merging
   })
