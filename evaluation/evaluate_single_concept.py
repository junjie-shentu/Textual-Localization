from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from utils.evaluation_metrics import CalculateMetrics

from ..utils.CustomModelLoader import CustomModelLoader

RAMDOM_SEED = [100, 101, 102, 103, 104]

def build_pipeline(ckpt_path):

    # Load the pipeline with the same arguments (model, revision) that were used for training
    model_id = "runwayml/stable-diffusion-v1-5"

    #specify the model type

    if "textual_inversion" in ckpt_path:
        pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        pipeline.load_textual_inversion(ckpt_path)

    elif "dreambooth" in ckpt_path:
        unet_path = os.path.join(ckpt_path, "unet")
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
        if os.path.exists(os.path.join(ckpt_path, "text_encoder")):
            text_encoder_path = os.path.join(ckpt_path, "text_encoder")
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)
            pipeline = StableDiffusionPipeline.from_pretrained(model_id,
                                                    unet=unet, 
                                                    text_encoder=text_encoder, 
                                                    torch_dtype=torch.float16, 
                                                    use_safetensors=True).to("cuda")

        else:
            pipeline = StableDiffusionPipeline.from_pretrained(model_id,
                                                            unet=unet,  
                                                            torch_dtype=torch.float16, 
                                                            use_safetensors=True).to("cuda")

    elif "custom_diffusion" in ckpt_path:
        if "dreambooth_test" in ckpt_path:
            #This is used for the custom diffusion model with cross attention guidance, comment if the ckpt is from the original custom diffusion model
            unet_path = os.path.join(ckpt_path, "unet")
            unet = UNet2DConditionModel.from_pretrained(ckpt_path, torch_dtype=torch.float16)
            pipeline = StableDiffusionPipeline.from_pretrained(model_id, unet=unet, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

        else:
            # #This is used for original custom diffusion model
            # pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
            # pipeline.unet.load_attn_procs(ckpt_path, weight_name="pytorch_custom_diffusion_weights.bin")

            #since now we are using our re-implemented custom diffusion model, we need to load the model by using the CustomModelLoader
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
            loader.load_attn_procs(ckpt_path, weight_name="pytorch_custom_diffusion_weights.bin", train_q=train_q, train_k=train_k, train_v=train_v, train_out=train_out)

        pipeline.load_textual_inversion(ckpt_path, weight_name="<new1>.bin")

    elif "textual_localization" in ckpt_path:
        # #This is used for original custom diffusion model
        # pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
        # pipeline.unet.load_attn_procs(ckpt_path, weight_name="pytorch_custom_diffusion_weights.bin")

        #since now we are using our re-implemented custom diffusion model, we need to load the model by using the CustomModelLoader
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
        loader.load_attn_procs(ckpt_path, weight_name="pytorch_custom_diffusion_weights.bin", train_q=train_q, train_k=train_k, train_v=train_v, train_out=train_out)

        pipeline.load_textual_inversion(ckpt_path, weight_name="<new1>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new2>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new3>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new4>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new5>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new6>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new7>.bin")
        # pipeline.load_textual_inversion(ckpt_path, weight_name="<new8>.bin")

    else:
        raise NotImplementedError("Please specify the correct ckpt_path")
    
    return pipeline


def generate_image(ckpt_path, text_prompt_list):

    #get the pipeline
    pipeline = build_pipeline(ckpt_path)

    all_generated_images = {}
    for text_prompt in tqdm(text_prompt_list, desc='Text Prompt Loop'):
        all_generated_images[text_prompt] = []
        for seed in tqdm(RAMDOM_SEED, desc='Seed Loop'):
            generator = torch.Generator("cuda").manual_seed(seed)
            images = pipeline(prompt=text_prompt, num_images_per_prompt=10, num_inference_steps=50, generator = generator).images #generate 10 images once, 50 images in total; return a list of PIL images
            all_generated_images[text_prompt].extend(images)

    return all_generated_images


def save_generated_images(all_generated_images, image_output_path):
    for text_prompt, image_list in all_generated_images.items():
        for i, image in enumerate(image_list):
            image.save(os.path.join(image_output_path, f"{text_prompt}_{i}.jpg"))



def evaluate_image(ckpt_path, text_prompt_list, real_folder_path, image_output_path):

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
        all_generated_images = generate_image(ckpt_path, text_prompt_list)#return a dict of list of PIL images for each text prompt: {"text_prompt": [PIL images]}
    # #generate the images
    # all_generated_images = generate_image(ckpt_path, text_prompt_list) #return a dict of list of PIL images for each text prompt: {"text_prompt": [PIL images]}

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

    NEW_TOKEN = "<new1>"#"sks"#"<new1>"#"ck"     if you are using multiple NEW_TOKENs, remember to load them in line 110

    object_name = "cat"

    model_version = "4_2"

    text_prompt_path = f"./text_prompt/{object_name}.txt"

    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        return [line.strip().format(NEW_TOKEN) for line in lines]

    text_prompt_list = read_text_file(text_prompt_path)

    real_folder_path = f"./my_dataset/single_object/{object_name}/resized"

    ckpt_path_list = [
                        # "/home2/jvqx51/lora_db/dreambooth_test/dreambooth_output/single_object/pet_cat5/checkpoint-50",
                        # "/home2/jvqx51/lora_db/dreambooth_test/dreambooth_output/single_object/pet_cat5/checkpoint-100",
                        # "/home2/jvqx51/lora_db/dreambooth_test/dreambooth_output/single_object/pet_cat5/checkpoint-150",
                        # "/home2/jvqx51/lora_db/dreambooth_test/dreambooth_output/single_object/pet_cat5/checkpoint-200",
                        # "/home2/jvqx51/lora_db/dreambooth_test/dreambooth_output/single_object/pet_cat5/checkpoint-400",

                        # "/home2/jvqx51/lora_db/dreambooth_test/custom_diffusion_output/single_object/w_ppl/pot(no_cross_attn_loss)/checkpoint-50",
                        # "/home2/jvqx51/lora_db/dreambooth_test/custom_diffusion_output/single_object/w_ppl/pot(no_cross_attn_loss)/checkpoint-100",
                        # "/home2/jvqx51/lora_db/dreambooth_test/custom_diffusion_output/single_object/w_ppl/pot(no_cross_attn_loss)/checkpoint-150",
                        # "/home2/jvqx51/lora_db/dreambooth_test/custom_diffusion_output/single_object/w_ppl/pot(no_cross_attn_loss)/checkpoint-200",
                        # "/home2/jvqx51/lora_db/dreambooth_test/custom_diffusion_output/single_object/w_ppl/pot(no_cross_attn_loss)/checkpoint-400",
                        # "/home2/jvqx51/lora_db/dreambooth_test/custom_diffusion_output/single_object/w_ppl/pot(no_cross_attn_loss)/checkpoint-600",

                        # "./output/dreambooth/w_ppl/pot/checkpoint-10",
                        # "./output/dreambooth/w_ppl/pot/checkpoint-20",
                        # "./output/dreambooth/w_ppl/pot/checkpoint-30",
                        # "./output/dreambooth/w_ppl/pot/checkpoint-40",
                        # "./output/dreambooth/w_o_ppl/pot/checkpoint-50",
                        # "./output/dreambooth/w_o_ppl/pot/checkpoint-100",
                        # "./output/dreambooth/w_o_ppl/pot/checkpoint-150",
                        # "./output/dreambooth/w_o_ppl/pot/checkpoint-200",

                        # "./output/textual_inversion/pot/learned_embeds-steps-200.safetensors",
                        # "./output/textual_inversion/pot/learned_embeds-steps-400.safetensors",
                        # "./output/textual_inversion/pot/learned_embeds-steps-600.safetensors",
                        # "./output/textual_inversion/pot/learned_embeds-steps-800.safetensors",
                        # "./output/textual_inversion/pot/learned_embeds-steps-1000.safetensors",

                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-10",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-20",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-30",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-40",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-50",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-100",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-150",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-200",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-250",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-300",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-350",
                        # "./output/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/checkpoint-400",

                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-10",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-20",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-30",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-40",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-50",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-100",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-150",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-200",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-250",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-300",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-350",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv/checkpoint-400",

                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-50",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-100",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-150",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-200",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-250",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-300",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-350",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(single_input)/checkpoint-400",

                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-50",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-100",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-150",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-200",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-250",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-300",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-350",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv(w_o_attnloss)/checkpoint-400",

                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-50",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-100",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-150",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-200",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-250",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-300",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-350",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwkwv/checkpoint-400",

                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-50",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-100",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-150",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-200",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-250",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-300",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-350",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wqwv/checkpoint-400",

                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-50",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-100",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-150",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-200",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-250",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-300",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-350",
                        f"/home2/jvqx51/lora_db/textual_localization/output/single_object/textual_localization/{object_name}_{model_version}/wkwv/checkpoint-400",

                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-10",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-20",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-30",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-40",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-50",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-100",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-150",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-200",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-250",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-300",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-350",
                        # "/home2/jvqx51/lora_db/textual_localization/output/single_object/custom_diffusion/pot_test3_1_wkwv(multi_input)/checkpoint-400",
                      ]

    #ckpt_path = "/home2/jvqx51/lora_db/dreambooth_test/dreambooth_output/single_object/pet_cat5/checkpoint-400"#"output/dreambooth/w_o_ppl/pet_cat5/checkpoint-150"

    image_output_path_list = [
                            # "./generated_images/textual_localization/pet_cat5(DB_1_1)/50steps",
                            # "./generated_images/textual_localization/pet_cat5(DB_1_1)/100steps",
                            # "./generated_images/textual_localization/pet_cat5(DB_1_1)/200steps",
                            # "./generated_images/textual_localization/pet_cat5(DB_1_1)/250steps",
                            # "./generated_images/textual_localization/pet_cat5(DB_1_1)/500steps",

                            # "./generated_images/custom_diffusion/w_ppl/pot(no_cross_sttn_loss)/50steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot(no_cross_sttn_loss)/100steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot(no_cross_sttn_loss)/150steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot(no_cross_sttn_loss)/200steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot(no_cross_sttn_loss)/400steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot(no_cross_sttn_loss)/600steps",

                            # "./generated_images/dreambooth/w_ppl/pot/10steps",
                            # "./generated_images/dreambooth/w_ppl/pot/20steps",
                            # "./generated_images/dreambooth/w_ppl/pot/30steps",
                            # "./generated_images/dreambooth/w_ppl/pot/40steps",
                            # "./generated_images/dreambooth/w_o_ppl/pot/50steps",
                            # "./generated_images/dreambooth/w_o_ppl/pot/100steps",
                            # "./generated_images/dreambooth/w_o_ppl/pot/150steps",
                            # "./generated_images/dreambooth/w_o_ppl/pot/200steps",

                            # "./generated_images/textual_inversion/pot/200steps",
                            # "./generated_images/textual_inversion/pot/400steps",
                            # "./generated_images/textual_inversion/pot/600steps",
                            # "./generated_images/textual_inversion/pot/800steps",
                            # "./generated_images/textual_inversion/pot/1000steps",

                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/10steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/20steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/30steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/40steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/50steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/100steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/150steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/200steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/250steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/300steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/350steps",
                            # "./generated_images/custom_diffusion/w_ppl/pot_test4_w_o_class_token_wvwkwout_denoise_wqwout_attention/400steps",

                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/10steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/20steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/30steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/40steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/50steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/100steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/150steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/200steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/250steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/300steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/350steps",
                            # "./generated_images/custom_diffusion/pot/pot_test3_1_wkwv/400steps",

                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/50steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/100steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/150steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/200steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/250steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/300steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/350steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv(single_input)/400steps",

                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/50steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/100steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/150steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/200steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/250steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/300steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/350steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}(w_o_attn_loss)/wkwv/400steps",

                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/50steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/100steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/150steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/200steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/250steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/300steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/350steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwkwv/400steps",

                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/50steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/100steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/150steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/200steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/250steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/300steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/350steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wqwv/400steps",

                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/50steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/100steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/150steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/200steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/250steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/300steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/350steps",
                            f"./generated_images/textual_localization/{object_name}/{object_name}_{model_version}/wkwv/400steps",

                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/10steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/20steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/30steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/40steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/50steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/100steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/150steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/200steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/250steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/300steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/350steps",
                            # "./generated_images/custom_diffusion/pot/pot_test_3_1_wkwv(multi_input)/400steps",
                        ]
    
    for ckpt_path, image_output_path in zip(ckpt_path_list, image_output_path_list):
        os.makedirs(image_output_path, exist_ok=True)

        result = evaluate_image(ckpt_path, text_prompt_list, real_folder_path, image_output_path)

        print(ckpt_path)
        print(result)





    



