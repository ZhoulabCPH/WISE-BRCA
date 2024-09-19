import numpy as np
import feather
import torch
import concurrent.futures
import os

from ParamNet.model import ParamNet
from PIL import Image


def norm(image):
    image = np.array(image).astype(np.float32)
    image = image.transpose((2, 0, 1))
    image = ((image / 255) - 0.5) / 0.5
    image = image[np.newaxis, ...]
    image = torch.from_numpy(image)
    return image


def un_norm(image):
    image = image.cpu().detach().numpy()[0]
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1, 2, 0))
    return image


def normalization(patch_path, slide_name):
    img_source_hist = Image.open(patch_path)
    img_norm_hist = model_hist(norm(img_source_hist))
    img_norm_hist = un_norm(img_norm_hist)
    img_norm_hist = Image.fromarray(img_norm_hist)
    img_norm_hist.save(f'{patches_path}/{slide_name}/{patch_path.split("/")[-1]}')


def process_normalization(slide_name):
    print(f'{slide_name} begin!')

    patches_name = feather.read_dataframe(f'{patches_path}/{slide_name}').index
    for patch_name in patches_name:
        patch_path = f'{patches_path}/{slide_name}/{patch_name}'
        normalization(patch_path, slide_name)
    print(f'{slide_name} finish!')



if __name__ == '__main__':

    model_hist = ParamNet()
    model_hist.load_state_dict(torch.load("checkpoints/ParamNet-Uni.pt")['net_G_A'])

    n_cluster = 30
    sample_num_224 = 90
    sample_num_512 = 60

    patches_path_224 = '../datasets/Patches_224'
    patches_path_512 = '../datasets/Patches_512'

    print(f'Patches 224 stain normalization begin')
    num_threads = 4
    patches_path = patches_path_224
    slides_name = os.listdir(patches_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_normalization, slides_name)

    print(f'Patches 512 stain normalization begin')
    num_threads = 4
    patches_path = patches_path_512
    slides_name = os.listdir(patches_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_normalization, slides_name)
