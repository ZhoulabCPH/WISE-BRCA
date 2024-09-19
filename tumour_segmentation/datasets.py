import torchvision.transforms as transforms
import torch
import os

from torch.utils.data import Dataset
from PIL import Image



class BC(Dataset):
    def __init__(self, workspace, patches_dir, transform=None):
        super(BC, self).__init__()
        self.workspace = workspace
        self.transform = transform
        self.patches_dir = patches_dir

    def __len__(self):
        return len(self.workspace)

    def load_patches(self, patch):
        dir = f'{self.patches_dir}/{patch}'
        patch = Image.open(f'{dir}.png').convert('RGB')
        patch = self.transform(patch)
        return patch

    def __getitem__(self, item):
        patch = self.workspace.iloc[item]['patches']
        purity = self.workspace.iloc[item]['tumor_percentage']
        if purity < 0.25:
            label = 0
        else:
            label = 1
        patch = self.load_patches(patch)
        return patch, label



class BC_inference(Dataset):
    def __init__(self, slide_path, transform=None,
                 ):
        super(BC_inference, self).__init__()
        self.images = list(os.listdir(slide_path))
        self.patches_dir = slide_path
        self.transform = transform

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        patch_dir = f'{self.patches_dir}/{image}'
        patch = Image.open(patch_dir).convert('RGB')
        patch = self.transform(patch)
        return patch, image

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1