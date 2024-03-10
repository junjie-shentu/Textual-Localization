from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse
from textual_localization.utils.utils.evaluation_metrics import CalculateMetrics

from textual_localization.utils.utils.CustomModelLoader import CustomModelLoader

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--RAMDOM_SEED_LOW', type=int, required=True, help='The random seed low bound')
parser.add_argument('--RAMDOM_SEED_HIGH', type=int, required=True, help='The random seed high bound')
parser.add_argument('--NEW_TOKEN', type=str, required=True, help='The new token')
parser.add_argument('--object_name', type=str, required=True, help='The object name')
parser.add_argument('--ckpt_path', type=str, required=True, help='The checkpoint path')
parser.add_argument('--image_output_path', type=str, required=True, help='The image output path')


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().format(NEW_TOKEN) for line in lines]

def build_pipeline(ckpt_path):

    # Load the pipeline with the same arguments (model, revision) that were used for training
    model_id = "runwayml/stable-diffusion-v1-5"

    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

    #register unet for loader
    if "wq" in ckpt_path:
        train_q = True
    else:
        train_q = False
    if "wk" in ckpt_path:
        train_k = True
    else:
        train_k = False
    if "wv" in ckpt_path:
        train_v = True
    else:
        train_v = False
    if "wout" in ckpt_path:
        train_out = True
    else:
        train_out = False
    
    loader = CustomModelLoader(pipeline.unet)
    loader.load_attn_procs(ckpt_path, weight_name="pytorch_textual_localization_weights.bin", train_q=train_q, train_k=train_k, train_v=train_v, train_out=train_out)

    pipeline.load_textual_inversion(ckpt_path, weight_name="<new1>.bin")

    
    return pipeline


def generate_image(ckpt_path, text_prompt_list):

    if RAMDOM_SEED_LOW == RAMDOM_SEED_HIGH:
        RAMDOM_SEED = [RAMDOM_SEED_LOW]
    else:
        RAMDOM_SEED = range(RAMDOM_SEED_LOW, RAMDOM_SEED_HIGH)

    #get the pipeline
    pipeline = build_pipeline(ckpt_path)

    all_generated_images = {}
    for text_prompt in tqdm(text_prompt_list, desc='Text Prompt Loop'):
        all_generated_images[text_prompt] = []
        for seed in tqdm(RAMDOM_SEED, desc='Seed Loop'):
            generator = torch.Generator("cuda").manual_seed(seed)
            images = pipeline(prompt=text_prompt, num_images_per_prompt=10, num_inference_steps=50, generator = generator).images #generate 10 images once, return a list of PIL images
            all_generated_images[text_prompt].extend(images)

    return all_generated_images


def save_generated_images(all_generated_images, image_output_path):
    for text_prompt, image_list in all_generated_images.items():
        for i, image in enumerate(image_list):
            image.save(os.path.join(image_output_path, f"{text_prompt}_{i}.jpg"))



def evaluate_image(ckpt_path, text_prompt_list, real_folder_path, image_output_path):

    os.makedirs(image_output_path, exist_ok=True)

    if len(os.listdir(image_output_path))> 0:
        #check if the generated images already exist
        #if yes, load the generated images
        all_generated_images = {}
        for text_prompt in text_prompt_list:
            all_generated_images[text_prompt] = []
            for filename in os.listdir(image_output_path):
                if filename.startswith(text_prompt):
                    image_path = os.path.join(image_output_path, filename)
                    with Image.open(image_path) as img:
                        all_generated_images[text_prompt].append(img.copy())
    
    else:
        #if not, generate the images
        all_generated_images = generate_image(ckpt_path, text_prompt_list, RAMDOM_SEED_LOW, RAMDOM_SEED_HIGH)#return a dict of list of PIL images for each text prompt: {"text_prompt": [PIL images]}

    #save the generated images
    save_generated_images(all_generated_images, image_output_path)

    #load the real images
    real_images = []
    for filename in os.listdir(real_folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.JPG'):
            image_path = os.path.join(real_folder_path, filename)
            with Image.open(image_path).convert('RGB') as img:
                real_images.append(img.copy())

    all_clip_i = {}
    all_clip_t = {}
    all_kid_score_mean = {}
    all_kid_score_std = {}
    all_lpips_score = {}

    for text_prompt, image_list in all_generated_images.items():
        text_prompt = text_prompt.replace(NEW_TOKEN, " ").strip()
        eval = CalculateMetrics(image_list, real_images, text_prompt)
        clip_i, clip_t, kid_score_mean, kid_score_std, lpips_score = eval()
        all_clip_i[text_prompt] = clip_i
        all_clip_t[text_prompt] = clip_t
        all_kid_score_mean[text_prompt] = kid_score_mean
        all_kid_score_std[text_prompt] = kid_score_std
        all_lpips_score[text_prompt] = lpips_score

    clip_i_mean = np.mean([tensor.cpu().numpy() for tensor in all_clip_i.values()])
    clip_t_mean = np.mean([tensor.cpu().numpy() for tensor in all_clip_t.values()])
    kid_score_mean_mean = np.mean([tensor.cpu().numpy() for tensor in all_kid_score_mean.values()])
    kid_score_std_mean = np.mean([tensor.cpu().numpy() for tensor in all_kid_score_std.values()])
    lpips_score_mean = np.mean([tensor.cpu().numpy() for tensor in all_lpips_score.values()])


    return {"clip_i_mean": clip_i_mean,
            "clip_t_mean": clip_t_mean,
            "kid_score_mean_mean": kid_score_mean_mean,
            "kid_score_std_mean": kid_score_std_mean,
            "lpips_score_mean": lpips_score_mean}


if __name__ == "__main__":

    args = parser.parse_args()

    NEW_TOKEN = args.NEW_TOKEN
    RAMDOM_SEED_LOW = args.RAMDOM_SEED_LOW
    RAMDOM_SEED_HIGH = args.RAMDOM_SEED_HIGH


    text_prompt_path = f"./text_prompt/{args.object_name}.txt"
    text_prompt_list = read_text_file(text_prompt_path)

    real_folder_path = f"../data/single_object/{args.object_name}/resized"


    result = evaluate_image(args.ckpt_path, text_prompt_list, real_folder_path, args.image_output_path)

    print(result)





    



