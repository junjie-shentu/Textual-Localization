# Textual Localization: Decomposing Multi-concept Images for Subject-Driven Text-to-Image Generation

Subject-driven text-to-image diffusion models empower users to tailor the model to new concepts absent in the pre-training dataset using a few sample images. However, prevalent subject-driven models primarily rely on single-concept input images, facing challenges in specifying the target concept when dealing with multi-concept input images. To this end, we introduce a textual localized text-to-image model (*Texual Localization*) to handle multi-concept input images. During fine-tuning, our method incorporates a novel cross-attention guidance to decompose multiple concepts, establishing distinct connections between the visual representation of the target concept and the identifier token in the text prompt. Experimental results reveal that our method outperforms or performs comparably to the baseline models in terms of image fidelity and image-text alignment on multi-concept input images. In comparison to *Custom Diffusion*, our method with hard guidance achieves CLIP-I scores that are 7.04%, 8.13% higher and CLIP-T scores that are 2.22%, 5.85% higher in single-concept and multi-concept generation, respectively. Notably, our method generates cross-attention maps consistent with the target concept in the generated images, a capability absent in existing models.

[Paper Link ](https://arxiv.org/abs/2402.09966)

## Getting Started
### setup
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