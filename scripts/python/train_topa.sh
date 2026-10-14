
toep_r=3
block_size=512

BASE_MODEL="meta-llama/Llama-3-8b-hf"
OUTPUT_PATH="output/python-Llama-3-8b"
DATA_PATH="pissa-dataset"

deepspeed --master_port=16955 --include=localhost:0,1,2,3 train_topa.py \
    --deepspeed configs/ds_config_zero2.json \
    --model_name_or_path $BASE_MODEL \
    --full_finetune False \
    --bf16 \
    --init_weights True \
    --target_modules "q_proj,v_proj" \
    --toep_r ${toep_r} \
    --block_size ${block_size} \
    --data_path $DATA_PATH \
    --sub_task python \
    --dataset_split train \
    --dataset_field instruction output \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --merge True \
    --use_gradient_checkpointing True
    

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m utils.gen_vllm --model $OUTPUT_PATH --sub_task python --output_file $OUTPUT_PATH/python_response.jsonl
python -m utils.code_process --path $OUTPUT_PATH/python_response.jsonl
evalplus.evaluate --dataset humaneval --samples $OUTPUT_PATH/humaneval.jsonl
evalplus.evaluate --dataset mbpp --samples $OUTPUT_PATH/mbpp.jsonl
