#!/bin/bash

LORA_PATH=$1
BASE_MODEL=$2
GPU_ID=$3
OUTPUT_PATH=$4

echo $LORA_PATH
echo $BASE_MODEL
echo $GPU_ID
echo $OUTPUT_PATH

export CUDA_VISIBLE_DEVICES=$GPU_ID

source $(conda info --base)/etc/profile.d/conda.sh
mkdir -p $OUTPUT_PATH

conda deactivate
conda activate lmeval

if [ "$LORA_PATH" = "None" ]; then
  echo "[INFO] Evaluating BASE model only"
  MODEL_ARGS="pretrained=$BASE_MODEL"
else
  echo "[INFO] Evaluating BASE + LoRA"
  MODEL_ARGS="pretrained=$BASE_MODEL,peft=$LORA_PATH"
fi

lm_eval --model hf \
 --model_args $MODEL_ARGS \
 --tasks gsm8k_cot \
 --limit 100 \
 --device cuda:$GPU_ID \
 --batch_size 16 \
 --output_path $OUTPUT_PATH \
 --seed 42

lm_eval --model hf \
   --model_args $MODEL_ARGS \
   --tasks m_mmlu_fr,arc_fr,hellaswag_fr,m_mmlu_es,arc_es,hellaswag_es,m_mmlu_de,arc_de,hellaswag_de,m_mmlu_ru,arc_ru,hellaswag_ru \
   --limit 100 \
   --device cuda:$GPU_ID \
   --batch_size 4 \
   --output_path $OUTPUT_PATH \
   --seed 42

lm_eval --model hf \
   --model_args $MODEL_ARGS \
   --tasks ifeval \
   --limit 100 \
   --device cuda:$GPU_ID \
   --batch_size 8 \
   --output_path $OUTPUT_PATH \
   --seed 42

 conda deactivate
 conda activate safety-eval
 cd safety-eval-fork

 export OPENAI_API_KEY=''

 LORA_ADAPTER_PATH="$OUTPUT_PATH"

 python evaluation/eval.py generators \
    --model_name_or_path "../$LORA_ADAPTER_PATH" \
    --base_model "$BASE_MODEL" \
    --tasks xstest \
    --model_input_template_path_or_name llama3 \
    --report_output_path ../$OUTPUT_PATH/safety_eval.json \
    --save_individual_results_path ../$OUTPUT_PATH/safety_generation.json \
    --batch_size 8


