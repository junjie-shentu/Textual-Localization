export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="path_to_instance_dir"
export HEATMAP_DIR_1="path_to_mask_dir (first_concept)"
export HEATMAP_DIR_2="path_to_mask_dir (second_concept)"
export CLASS_DIR_1="path_to_prior_preservation_image_dir (first_concept)"
export CLASS_DIR_2="path_to_prior_preservation_image_dir (second_concept)"
export OUTPUT_DIR="path_to_output_dir"
export CLASS_PROMPT_1="class_name_of_the_target_concept (first_concept)"
export CLASS_PROMPT_2="class_name_of_the_target_concept (second_concept)"
export INSTANCE_PROMPT="photo of a <new1> [class_name_1] and a <new2> [class_name_2] (e.g., pot, penbag, bucket, doll etc.)"

accelerate launch ./model/multi_concept/text_localization_hard_guidance.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --heatmap_dir_1=$HEATMAP_DIR_1 \
  --heatmap_dir_2=$HEATMAP_DIR_2 \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir_1=$CLASS_DIR_1 \
  --class_data_dir_2=$CLASS_DIR_2 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt_1=$CLASS_PROMPT_1 \
  --class_prompt_2=$CLASS_PROMPT_2 \
  --num_class_images=200 \
  --instance_prompt=$INSTANCE_PROMPT  \
  --resolution=512  \
  --train_batch_size=3  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --scale_lr \
  --modifier_token="<new1>+<new2>" \
  --initializer_token="sks+sks" \
  --report_to="wandb" \
  --no_safe_serialization \
  --checkpointing_steps=100 \
  --noaug \
  --train_v --train_k \
