import torch
import torchvision.transforms as transforms
import numpy as np
import clip
import torchvision
from PIL import Image
import os

def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for i, classname in enumerate(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).detach().cuda()
    return zeroshot_weights

def img_process(images, img_size):
    roiAlign = torchvision.ops.RoIAlign(output_size=224, sampling_ratio = -1, spatial_scale=1, aligned = True)
    batch_image = []
    coord = torch.tensor([[0.0,0.0,float(img_size),float(img_size)]]).cuda().to(torch.float16)
    for i in range(images.shape[0]):
        image = images[i].unsqueeze(0)
        image = roiAlign(image, [coord]).squeeze()
        batch_image.append(image)
    batch_image = torch.stack(batch_image, dim=0)
    return batch_image



def save_pil_image(image, clean_text, successful, adv_text):

    file_path = os.path.join('images',clean_text, successful)
    
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    adv_text = clean_filename(adv_text)
    save_path = os.path.join(file_path,adv_text+'.png')

    image[0].save(save_path)
    #print(save_path)


import string

def clean_filename(filename):
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned_filename = "".join(c for c in filename if c in valid_chars)
    return cleaned_filename

