export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="path_to_instance_dir"
export MASK_DIR_1="path_to_mask_dir"
export CLASS_DIR_1="path_to_prior_preservation_image_dir"
export OUTPUT_DIR="path_to_output_dir"
export CLASS_PROMPT_1="class_name_of_the_target_concept"
export INSTANCE_PROMPT="photo of a <new1> [class_name] (e.g., pot, penbag, bucket, doll etc.)"

accelerate launch ./model/single_object/text_localization_hard_guidance.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --heatmap_dir_1=$MASK_DIR_1 \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir_1=$CLASS_DIR_1 \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --class_prompt_1=$CLASS_PROMPT_1 \
  --num_class_images=200 \
  --instance_prompt=$INSTANCE_PROMPT  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --scale_lr  \
  --modifier_token="<new1>" \
  --initializer_token="sks" \
  --report_to="wandb" \
  --no_safe_serialization \
  --checkpointing_steps=50 \
  --noaug \
  --train_v --train_k \
