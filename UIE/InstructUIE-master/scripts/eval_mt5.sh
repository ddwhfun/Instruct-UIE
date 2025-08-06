#!/bin/bash
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

port=$(shuf -i25000-30000 -n1)

# 最好令 bs_per_gpu * num_gpu * gradient_accumulation_steps = 256
# 学习率可以使用 5e-5
# param_num < 1b 10epoch, 3b 5epoch, 11b 5epoch
# 注意修改 CUDA_VISIBLE_DEVICES, model_name_or_path，output_dir, run_name, data_dir, task_config_dir, instruction_file
# 其余参数可与当前版本保持一致

# 3090 * 4 on t5-700M

output_dir=output/eval-mt5-base-clw-ckp9600


deepspeed --include localhost:1 --master_port $port src/run_uie.py \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/train-mt5-base-clw/checkpoint-9600 \
   --data_dir ../dataset_new \
   --output_dir $output_dir \
   --task_config_dir ../multitasks_configs_new \
   --instruction_file configs/instruction_config_clw.json \
   --instruction_strategy single \
   --input_record_file flan-t5.record \
   --per_device_eval_batch_size 16 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name t5-700M-mult-mi-experiment \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --max_num_instances_per_eval_task 200 \
   --add_task_name False \
   --add_dataset_name False \
   --num_examples 0 \
   --overwrite_output_dir \
   --overwrite_cache