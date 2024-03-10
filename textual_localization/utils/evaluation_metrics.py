import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchxrayvision as xrv
import numpy as np
from skimage.color import gray2rgb
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor, CLIPVisionModel
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity



class XRVDatabase(Dataset):
    def __init__(self, images_list):
        super().__init__()
        self.images_list = images_list
        self.transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])


    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        #image_path = self.data_file_path + self.disease_name + "/" + "train_" + str(idx) + ".png" #"./data/generated_image_sd/" + self.disease_name + "/" + "train_" + str(idx) + ".png"
        #img = skimage.io.imread(self.images_list[idx])
        img = np.asarray(self.images_list[idx])# equals to skimage.io.imread(path of image)
        if img.ndim == 2:
                img = gray2rgb(img)
        
        img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
        img = img.mean(2)[None, ...] # Make single color channel
        img = self.transform(img)
        img = torch.from_numpy(img)

        return img

def collate_fn(batch):
    return torch.stack(batch)

class CalculateMetrics(nn.Module):#pil images in list
    """Calculate the metrics for the generated images"""
    def __init__(self, generated_images, real_images, text_prompt):
        super().__init__()
        self.generated_images = generated_images
        self.real_images = real_images

        self.clipmodel = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clipvisionmodel = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

        self.clipmodel.eval()
        self.clipvisionmodel.eval()

        self.text_prompt = text_prompt

    def image_transform(self, image, size, if_normalize):
        if if_normalize:
            image_transform = transforms.Compose(
                [
                    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),#[0,1]
                    transforms.Normalize([0.5],[0.5])#[-1,1]
                ])
        else:
            image_transform = transforms.Compose(
                [
                    transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()#[0,1]
                ])
        image_tensor = image_transform(image)

        return image_tensor

    def calculate_clip_i(self):
        with torch.no_grad():
            #calculate CLIP-I: CLIP embedding similarity between generated images and real images
            generated_image_tensor = self.processor(images=self.generated_images, return_tensors="pt").to("cuda")
            real_image_tensor = self.processor(images=self.real_images, return_tensors="pt").to("cuda")
            generated_image_embed = self.clipvisionmodel(**generated_image_tensor).pooler_output.to("cuda")
            real_image_embed = self.clipvisionmodel(**real_image_tensor).pooler_output.to("cuda")
            # compute pairwise cosine similarities, cos_sim tensor has a shape of (50, 5)
            # cos_sim[i, j] contains the cosine similarity between the i-th embedding in generated_image_embed and the j-th embedding in real_image_embed.
            cos_sim = F.cosine_similarity(generated_image_embed.unsqueeze(1), real_image_embed.unsqueeze(0), dim=-1).to("cuda")
            clip_i = torch.mean(cos_sim).to("cuda")

            return clip_i

    def calculate_clip_t(self, text_prompt):
        with torch.no_grad():
            #calculate CLIP-T: CLIP embedding similarity between generated images and text prompt
            inputs = self.processor(text=[text_prompt],images=self.generated_images, return_tensors="pt").to("cuda")#text=[text_prompt] * len(self.generated_images)
            outputs = self.clipmodel(**inputs)
            generated_image_embed = outputs.image_embeds.to("cuda")
            text_embed =  outputs.text_embeds.to("cuda")
            cos_sim = F.cosine_similarity(generated_image_embed.unsqueeze(1), text_embed.unsqueeze(0), dim=-1).to("cuda")
            clip_t = torch.mean(cos_sim).to("cuda")

            return clip_t
        
    def calculate_kid(self):
        with torch.no_grad():
            generated_image_tensor = torch.stack([self.image_transform(image, 299, False) for image in self.generated_images]).to("cuda")
            real_image_tensor = torch.stack([self.image_transform(image, 299, False) for image in self.real_images]).to("cuda")
            kid = KernelInceptionDistance(subset_size=5, normalize=True).to("cuda")#Argument `subset_size` should be smaller than the number of samples
            # print("generated_image_tensor_shape:", generated_image_tensor.shape)
            # print("real_image_tensor_shape:", real_image_tensor.shape)
            kid.update(real_image_tensor, real=True)
            kid.update(generated_image_tensor, real=False)
            kid_score_mean, kid_score_std = kid.compute()

            return kid_score_mean, kid_score_std
        
    # def calculate_xrv(self):
    #     with torch.no_grad():
    #         model = xrv.models.DenseNet(weights="densenet121-res224-all").to("cuda")
    #         dataloader_real = DataLoader(XRVDatabase(self.real_images), batch_size=20, shuffle=False, collate_fn=collate_fn)
    #         dataloader_generated = DataLoader(XRVDatabase(self.generated_images), batch_size=20, shuffle=False, collate_fn=collate_fn)

    #         real_features = []
    #         generated_features = []
    #         for batch_real, batch_generated in zip(dataloader_real, dataloader_generated):
    #             batch_real, batch_generated = batch_real.to("cuda"), batch_generated.to("cuda")
    #             xrv_feature_real = model.features2(batch_real)
    #             xrv_feature_generated = model.features2(batch_generated)
    #             real_features.append(xrv_feature_real)
    #             generated_features.append(xrv_feature_generated)

    #         real_features = torch.cat(real_features, dim=0)
    #         generated_features = torch.cat(generated_features, dim=0)

    #         xrv_score = F.cosine_similarity(real_features.unsqueeze(1), generated_features.unsqueeze(0), dim=-1).to("cuda")
    #         xrv_score = torch.mean(xrv_score).to("cuda")

    #         return xrv_score
        
    def calculate_lpips(self):
        with torch.no_grad():
            generated_image_tensor = torch.stack([self.image_transform(image, 256, False) for image in self.generated_images]).to("cuda")
            lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")
            lpips_scores = []
            for i in range(generated_image_tensor.shape[0]):
                for j in range (i+1, generated_image_tensor.shape[0]):
                    img1 = generated_image_tensor[i].unsqueeze(0).to("cuda")
                    img2 = generated_image_tensor[j].unsqueeze(0).to("cuda")
                    lpips_score = lpips(img1, img2)
                    lpips_scores.append(lpips_score)

            lpips_score = torch.mean(torch.stack(lpips_scores)).to("cuda")

            return lpips_score
            
    def forward(self):
        clip_i = self.calculate_clip_i()
        clip_t = self.calculate_clip_t(text_prompt = self.text_prompt)
        kid_score_mean, kid_score_std = self.calculate_kid()
        #xrv_score = self.calculate_xrv()
        lpips_score = self.calculate_lpips()

        return clip_i, clip_t, kid_score_mean, kid_score_std, lpips_score