from PIL import Image
from torchvision import transforms

""" Taken from the original ALBEF """
def preprocess_images(config, images, model_name):
    if(model_name == 'ALBEF' or model_name == 'XVLM' or model_name== 'BLIP' or model_name == 'X2VLM'):
        images = images.convert('RGB')
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

    return transform(images)