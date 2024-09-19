import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np

from model import WISE_BRCA_combined
from mcVAE.datasets import BC_multi_modal, collate_multi_modal
from torch.utils.data import DataLoader
from torch import nn



def pgBRCA_mutation_predict_multi_modal():
    slides_path_224 = rf'../datasets/WSIs_CTransPath_cluster_sample_224'
    slides_path_512 = rf'../datasets/WSIs_CTransPath_cluster_sample_512'
    phenotype_data_path = rf'../datasets/clinical_data/phenotype_data.csv'
    model = WISE_BRCA_combined().cuda()
    ckpt = torch.load(
        rf'../checkpoints/WISE-BRCA.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])

    slides_name = os.listdir(slides_path_224)
    labels = ['' for i in range(len(slides_name))]

    workspace = pd.DataFrame()
    workspace['slides'] = slides_name
    workspace['BRCA_mut'] = labels

    phenotype_data = pd.read_csv(phenotype_data_path)
    columns = ['SLIDES', 'BRCA_mut', 'age_at_diagnosis',
               'tumor_history', 'BRCA_history', 'OV_history', 'tumor_family_history',
               'BRCA_family_history', 'OV_family_history',
               'pancreatic_cancer_family_history', 'mbc_cancer_family_history',
               'largest_diameter', 'Grade', 'AR_grade', 'ER_grade', 'PR_grade', 'Ki67',
               'CK56', 'Lymph_node_status', 'HER2_0', 'HER2_1', 'multifocal_1',
               'multifocal_2']
    phenotype_data = phenotype_data.loc[:, columns]
    workspace = pd.concat(workspace, phenotype_data)
    data_inference = BC_multi_modal(workspace, slides_path_224, slides_path_512)
    data_inference_loader = DataLoader(data_inference, 16, shuffle=False, num_workers=4, drop_last=False,
                                   collate_fn=collate_multi_modal)
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        data = pd.DataFrame()
        model.eval()
        id = []
        score = np.array([])
        label = []
        for step, slide in enumerate(data_inference_loader):
            patches_224, patches_name_224, patches_512, patches_name_512, phenotype_feature, batch_labels, id_ = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['phenotype_feature'], slide['labels'], slide['id']
            id = id + id_
            pred_ = model.forward(patches_224.cuda(), patches_512.cuda())

            score = np.append(score, (sigmoid(pred_).detach().cpu().numpy()))

            label = label + list(batch_labels.detach().cpu().numpy())

        data['id'] = list(id)
        data['label'] = list(label)
        data['score'] = list(score)
    return data




















