# HF_HOME=/workspace/hf_Home OMP_NUM_THREADS=12 accelerate launch --config_file=/workspace/llm_training/deepspeed_zero2.yaml train.py --dataset_name lightblue/multi_context_closed_qa  --model_name mistralai/Mistral-7B-Instruct-v0.2 --dataset_col_name closedqa_messages --num_epochs 3 --train_batch_size 1 --eval_batch_size 4 --lr_scheduler_type cosine --optimizer adamw_8bit --neftune_noise_alpha 5.0 --run_name mistral7B0.2_long_closed_qa --do_lora True


HF_HOME=/workspace/hf_Home python train_trl.py \
    --model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2" \
    --report_to="all" \
    --learning_rate=2e-5 \
    --dataset_name lightblue/multi_context_closed_qa \
    --max_seq_length 2000 \
    --auto_find_batch_size False \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --output_dir="mistral7B0.2_long_closed_qa" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --evaluation_strategy="epoch" \
    --save_strategy="epoch" \
    --max_steps=-1 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=32 \
    --lora_alpha=16 \
    --bf16

