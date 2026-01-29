
# METIS: Many-shot Loss-Gap-Aware Model Merging

This repository provides the official implementation of **METIS**, a many-shot model merging framework designed to mitigate task interference and information erasure in large language models.

---

## Environment Setup


### 1. Create and activate a virtual environment
```bash
conda create -n metis python=3.10 -y
conda activate metis
````

### 2. Install dependencies

```bash
pip install -r metis_requirements.txt
```

(Optional) For evaluation:

```bash
pip install -r lmeval_requirements.txt
pip install -r safety-eval_requirements.txt
```

---

## Running METIS

The main entry point is `main_proposed.py`, which uses `fire` for command-line arguments.

```bash
python main_proposed.py run_balanced_model_merging \
  --global_model meta-llama/Llama-3.2-3B \
  --data_path ./data \
  --output_dir ./lora-model_ \
  --num_clients 4 \
  --num_communication_rounds 5
```

All arguments and default values are defined in the `balanced_model_merging()` function in `main_proposed.py`.

---

## Outputs

Merged LoRA adapters are saved under:

```text
lora-model_/<num_clients>/<round>/adapter_model_round<r>.bin
```

The final merged adapter is also saved as:

```text
lora-model_/<num_clients>/adapter_model.bin
```

---

## Evaluation

To evaluate the merged models on multiple benchmarks:

```bash
bash evaluate_all.sh
```
