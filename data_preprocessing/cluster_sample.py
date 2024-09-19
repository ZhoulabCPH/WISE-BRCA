import os

import feather

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import pandas as pd
import numpy as np
import random
import shutil
import concurrent.futures
from sklearn.cluster import KMeans
from torch import nn
import torchvision.transforms as transforms
from model import MossTumor
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    # cudnn.benchmark = False
    # cudnn.enabled = False


def cluster_sample_CTransPath(CTransPath_feature_path, tumour_segment_path, save_dir, n_cluster, sample_num):
    seed = 33
    setup_seed(seed)
    CTransPath_feature_path = CTransPath_feature_path
    tumor_segment_path = tumour_segment_path
    save_dir = save_dir
    count = 0
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for slide_name in os.listdir(CTransPath_feature_path):
        count = count + 1
        print(f'{count}-th {slide_name} begin!')
        if slide_name in os.listdir(save_dir):
            print(f'{count}-th {slide_name} already finish!')
            continue
        slide_resnet_features = feather.read_dataframe(f'{CTransPath_feature_path}/{slide_name}')
        if len(slide_resnet_features) == 0:
            print(f'Warning {slide_name}')
            continue

        slide_tumor_segmentation = feather.read_dataframe(f'{tumor_segment_path}/{slide_name}')

        slide_tumor_region = slide_tumor_segmentation[slide_tumor_segmentation.iloc[:, 1] == 1]
        if len(slide_tumor_region) < sample_num:
            slide_tumor_segmentation = slide_tumor_segmentation.sort_values(by=2, ascending=False)
            if len(slide_tumor_segmentation) < sample_num:
                random_select_ = np.random.choice(len(slide_tumor_segmentation), size=sample_num, replace=True)
                patches = slide_tumor_segmentation.iloc[random_select_, 0].values
            else:
                patches = slide_tumor_segmentation.iloc[0:sample_num, 0].values
            slide_features = slide_resnet_features.loc[patches]
            feather.write_dataframe(slide_features, f'{save_dir}/{slide_name}')
        else:
            patches = slide_tumor_region.iloc[:, 0].to_list()
            features = slide_resnet_features.loc[patches].values[:, 0:]
            kmeans = KMeans(n_clusters=n_cluster)
            kmeans.fit(features)
            cluster_labels = kmeans.labels_
            patch_cluster = {}
            for i in range(len(cluster_labels)):
                patch = patches[i]
                cluster = cluster_labels[i]
                if cluster not in patch_cluster.keys():
                    patch_cluster[cluster] = [patch]
                else:
                    patch_cluster[cluster].append(patch)
            candidate_patches = []
            for cluster in patch_cluster.keys():
                random.shuffle(patch_cluster[cluster])
                if len(patch_cluster[cluster]) >= sample_num / n_cluster:
                    for i in range(0, int(sample_num / n_cluster)):
                        candidate_patches.append(patch_cluster[cluster][i])
                else:
                    for i in range(0, int(sample_num / n_cluster)):
                        candidate_patches.append(patch_cluster[cluster][0])
            slide_features = slide_resnet_features.loc[candidate_patches]
            feather.write_dataframe(slide_features, f'{save_dir}/{slide_name}')


if __name__ == '__main__':
    n_cluster_224 = 30
    n_cluster_512 = 30

    n_sample_224 = 90
    n_sample_512 = 60

    WSIs_CTransPath_224_path = rf'../datasets/WSIs_CTransPath_224'
    WSIs_CTransPath_512_path = rf'../datasets/WSIs_CTransPath_512'

    WSIs_tumour_224_path = rf'../datasets/WSIs_tumour_224'
    WSIs_tumour_512_path = rf'../datasets/WSIs_tumour_512'

    WSIs_CTransPath_cluster_sample_224_path = rf'../datasets/WSIs_CTransPath_cluster_sample_224'
    WSIs_CTransPath_cluster_sample_512_path = rf'../datasets/WSIs_CTransPath_cluster_sample_512'

    print(f'Size 224 n_clusters: {n_cluster_224}, n_sample: {n_sample_224} begin!')
    cluster_sample_CTransPath(CTransPath_feature_path=WSIs_CTransPath_224_path,
                              tumour_segment_path=WSIs_tumour_224_path,
                              save_dir=WSIs_CTransPath_cluster_sample_224_path,
                              n_cluster=n_cluster_224, sample_num=n_sample_224)

    print(f'Size 512 n_clusters: {n_cluster_512}, n_sample: {n_sample_512} begin!')
    cluster_sample_CTransPath(CTransPath_feature_path=WSIs_CTransPath_512_path,
                              tumour_segment_path=WSIs_tumour_512_path,
                              save_dir=WSIs_CTransPath_cluster_sample_512_path,
                              n_cluster=n_cluster_512, sample_num=n_sample_512)

















