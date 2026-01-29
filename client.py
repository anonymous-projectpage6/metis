import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import set_seed as hf_set_seed
hf_set_seed(1234)

class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir):
        self.client_id = client_id
        self.model = model
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(
            self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id)
        )

    def preprare_local_dataset(self, preprocess_wrapper, round):
        all_examples = [ex for ex in self.local_data["train"]]
        debug_dir = os.path.join(
            self.output_dir,
            "input_prompt"
        )
        self.local_train_dataset = (
            self.local_data["train"]
            .shuffle(seed=1234)
            .map(lambda ex: preprocess_wrapper(
                ex,
                all_examples,
                round=round,
                client_id=self.client_id,
                output_dir=debug_dir
            ))
        )

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length):

        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            seed=1234,
            data_seed=1234,
            eval_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            dataloader_num_workers=0,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )

        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            ),
        )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict(
                (name, param.detach())
                for name, param in self.model.named_parameters()
                if "default" in name
            )
        )
        self.params_dict_new = OrderedDict(
            (name, param.detach())
            for name, param in self.model.named_parameters()
            if "default" in name
        )
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance,
                self.params_dict_new,
                "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, round, local_dataset_len_dict, previously_selected_clients_set):
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(
            self.output_dir, str(round), "local_output_{}".format(self.client_id)
        )
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")
        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
