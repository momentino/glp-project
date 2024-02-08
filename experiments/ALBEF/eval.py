from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import torch

""" Taken from the original ALBEF """
def preprocess_images(config, images):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    return transform(images)

def eval(model, loader, config):
    model.eval()
    with torch.no_grad():
        for images, captions, foils, categories in tqdm(loader):
            images = preprocess_images(config, images)
