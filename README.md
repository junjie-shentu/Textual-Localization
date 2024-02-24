# Textual Localization: Decomposing Multi-concept Images for Subject-Driven Text-to-Image Generation

Subject-driven text-to-image diffusion models empower users to tailor the model to new concepts absent in the pre-training dataset using a few sample images. However, prevalent subject-driven models primarily rely on single-concept input images, facing challenges in specifying the target concept when dealing with multi-concept input images. To this end, we introduce a textual localized text-to-image model (*Texual Localization*) to handle multi-concept input images. During fine-tuning, our method incorporates a novel cross-attention guidance to decompose multiple concepts, establishing distinct connections between the visual representation of the target concept and the identifier token in the text prompt. Experimental results reveal that our method outperforms or performs comparably to the baseline models in terms of image fidelity and image-text alignment on multi-concept input images. In comparison to *Custom Diffusion*, our method with hard guidance achieves CLIP-I scores that are 7.04%, 8.13% higher and CLIP-T scores that are 2.22%, 5.85% higher in single-concept and multi-concept generation, respectively. Notably, our method generates cross-attention maps consistent with the target concept in the generated images, a capability absent in existing models.

[Paper Link ](https://arxiv.org/abs/2402.09966)

## Getting Started
Download and set up the repo:
```
git clone https://github.com/junjie-shentu/Textual-Localization.git
cd Textual-Localization
```

Intsall environment:
```
conda create --name textual-localization --file environment.yml
conda activate textual-localization
```

## Training Texual Localization
### Learn single concept from multi-concept input images
Train the textual localization model to learn single concept from the input image, first modify the `train_TL_hard_single_concept.sh`/`train_TL_soft_single_concept.sh` to specify the variabls:
```
#take the class "doll" as an example
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/multi_concept/bucket+doll/resized"
export MASK_DIR_1="../data/multi_concept/bucket+doll/mask_2"
export CLASS_DIR_1="../data/prior_data/doll"
export OUTPUT_DIR="../output/single_concept/doll"
export CLASS_PROMPT_1="doll"
export INSTANCE_PROMPT="photo of a <new1> doll"
```

Then run the following command:
```
bash train_TL_hard_single_concept.sh
```
Or
```
bash train_TL_soft_single_concept.sh
```

### Learn multiple concepts from multi-concept input images
Train the textual localization model to learn multiple concepts from the input image, first modify the `train_TL_hard_multi_concept.sh`/`train_TL_soft_multi_concept.sh` to specify the variabls:
```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="../data/multi_concept/bucket+doll/resized"
export MASK_DIR_1="../data/multi_concept/bucket+doll/mask_1"
export MASK_DIR_2="../data/multi_concept/bucket+doll/mask_2"
export CLASS_DIR_1="../data/prior_data/bucket"
export CLASS_DIR_2="../data/prior_data/doll"
export OUTPUT_DIR="../output/multi_concept/bucket+doll"
export CLASS_PROMPT_1="bucket"
export CLASS_PROMPT_2="doll"
export INSTANCE_PROMPT="photo of a <new1> bucket and a <new2> doll"
```

Then run the following command:
```
bash train_TL_hard_multi_concept.sh
```
Or
```
bash train_TL_soft_multi_concept.sh
```

## Evaluation
Evaluate the textual localization model when learning single concept from multi-concept input images:
```
python .evaluation/evaluate_single_concept.py --RAMDOM_SEED_LOW 1 --RAMDOM_SEED_HIGH 6 --NEW_TOKEN "<NEW1>" --object "doll" --ckpt_path "../output/single_concept/doll/wkwv/checkpoint-100" --image_output_path "../generated_images/single_concept/doll"
```

Evaluate the textual localization model when learning multiple concepts from multi-concept input images:
```
python .evaluation/evaluate_multi_concept.py --RAMDOM_SEED_LOW 1 --RAMDOM_SEED_HIGH 6 --NEW_TOKEN_1 "<NEW1>" --NEW_TOKEN_2 "<NEW2>" --object_1 "bucket" --object_2 "doll" --ckpt_path "../output/multi_concept/bucket+doll/wkwv/checkpoint-100" --image_output_path "../generated_images/multi_concept/bucket+doll"
```

## Citation
If you find this work helpful, please consider citing the following BibTeX entry:
```
@article{shentu2024textual,
  title={Textual Localization: Decomposing Multi-concept Images for Subject-Driven Text-to-Image Generation},
  author={Shentu, Junjie and Watson, Matthew and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2402.09966},
  year={2024}
}
```