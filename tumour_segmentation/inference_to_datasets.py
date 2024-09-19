import torch
import os
import feather
import pandas as pd

from torch import nn
from model import Tumour_segmentation
from torch.utils.data import Dataset, DataLoader
from datasets import BC_inference, Transform





os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def inference(model, slides_path, save_path):

    with torch.no_grad():
        for slide_name in os.listdir(slides_path):
            softmax = nn.Softmax(dim=1)
            print(f'{slide_name} begin!')
            if f'{slide_name}.feather' not in os.listdir(save_path):

                data = BC_inference(slide_path=slides_path, transform=Transform())
                data_loader = DataLoader(data, 512, shuffle=False, num_workers=4, drop_last=False)
                data = pd.DataFrame()
                for i, (features, images) in enumerate(data_loader):
                    features = features.to(torch.device('cuda:0'))
                    score = model(features)
                    score = softmax(score)
                    score_ = [score[i][1].item() for i in range(len(score))]
                    pred_label = list(score.argmax(dim=1).detach().cpu().numpy())

                    images = list(images)
                    if len(images) != len(pred_label):
                        print('Warning!')
                    images_labels = [[images[i]] + [pred_label[i]] + [score_[i]] for i in range(len(images))]
                    data_ = pd.DataFrame(data=images_labels)
                    data = pd.concat([data, data_])

                feather.write_dataframe(data, rf'{save_path}/{slide_name}.feather')
        print(f'{slide_name} down!')


if __name__ == '__main__':
    model = Tumour_segmentation()
    ckpt = torch.load(
        rf'../checkpoints/Tumour_segmentation_model_224.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    slides_path = '../datasets/Patches_224'
    save_path = rf'../datasets/WSIs_tumour_224'
    inference(model, slides_path, save_path)