import torch
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from model import WISE_BRCA_combined
from mcVAE.datasets import BC_multi_modal, collate_multi_modal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
def get_args():
    parser = argparse.ArgumentParser(description='BRCA mutation prediction')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=60, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--workspace_CHCAMS_train_validation',
                        default=f'../datasets/clinical_data/CHCAMS_data_train_validation.csv')
    parser.add_argument('--workspace_CHCAMS_test', default=f'../datasets/clinical_data/CHCAMS_data_test.csv')
    parser.add_argument('--workspace_YYH', default=f'../datasets/clinical_data/YYH_data.csv')
    parser.add_argument('--CHCAMS_slides_path_224',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_224_CHCAMS')
    parser.add_argument('--CHCAMS_slides_path_512',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_512_CHCAMS')
    parser.add_argument('--YYH_slides_path_224',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_224_YYH')
    parser.add_argument('--YYH_slides_path_512',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_512_YYH')
    parser.add_argument('--checkpoint-path',
                        default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('--wd', default=5e-4)
    parser.add_argument('--T_max', default=10)
    args = parser.parse_args()
    return args

def calculate_metrics(model,data_loader):
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        data = pd.DataFrame()
        model.eval()
        id = []
        score = np.array([])
        label = []
        for step, slide in enumerate(data_loader):
            patches_224, patches_name_224, patches_512, patches_name_512, phenotype_feature, batch_labels, id_ = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['phenotype_feature'], slide['labels'], slide['id']
            id = id+id_
            pred_= model.forward(patches_224.cuda(), patches_512.cuda(), phenotype_feature.cuda())

            score = np.append(score, (sigmoid(pred_).detach().cpu().numpy()))

            label = label+list(batch_labels.detach().cpu().numpy())

        auc = roc_auc_score(label, score)
        data['id'] = list(id)
        data['label'] = list(label)
        data['score'] = list(score)

    return auc, data


def train():
    args = get_args()
    workspace_CHCAMS_train_validation = pd.read_csv(args.workspace_CHCAMS_train_validation)
    workspace_CHCAMS_test = pd.read_csv(args.workspace_CHCAMS_test)
    workspace_YYH = pd.read_csv(args.workspace_YYH)

    seed = 11
    setup_seed(seed)

    l_bce = nn.BCEWithLogitsLoss()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = WISE_BRCA_combined(head_hidden_1=1536, head_hidden_2=768, head_hidden_3=384, dropout=0.40).cuda()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(
        rf'../checkpoints/mcVAE.pth',
        map_location='cuda:0')['model']
    state_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    for k, v in model.named_parameters():
        if 'head' in k:
            continue
        else:
            v.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=20)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=100, after_scheduler=scheduler)
    data_train_validation = BC_multi_modal(workspace_CHCAMS_train_validation, f'{args.CHCAMS_slides_path_224}', f'{args.CHCAMS_slides_path_512}')
    data_train_validation_loader = DataLoader(data_train_validation, args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                           collate_fn=collate_multi_modal)

    data_test = BC_multi_modal(workspace_CHCAMS_test, f'{args.CHCAMS_slides_path_224}', f'{args.CHCAMS_slides_path_512}')
    data_test_loader = DataLoader(data_test, args.batch_size, shuffle=True, num_workers=4, drop_last=False,
                                  collate_fn=collate_multi_modal)

    data_YYH = BC_multi_modal(workspace_YYH, args.YYH_slides_path_224, args.YYH_slides_path_512)
    data_YYH_loader = DataLoader(data_YYH, args.batch_size, shuffle=True, num_workers=4, drop_last=False,
                           collate_fn=collate_multi_modal)

    early_stop = 0
    auc_test = 0
    epoch_test = 0
    for epoch in range(args.epochs):
        early_stop = early_stop + 1
        if early_stop > 10:
            print('Early stop!')
            break
        LOSS = []
        model.train()
        progress_bar = tqdm(total=len(data_train_validation_loader), desc="WISE_BRCA_combined training")
        for step, slide in enumerate(data_train_validation_loader, start=epoch * len(data_train_validation_loader)):
            patches_224, patches_name_224, patches_512, patches_name_512, phenotype_feature, batch_labels, id = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['phenotype_feature'], slide['labels'], slide['id']

            optimizer.zero_grad()

            pred = model.forward(patches_224.cuda(), patches_512.cuda(), phenotype_feature.cuda())
            loss = l_bce(pred.to(torch.float64), batch_labels.cuda().to(torch.float64))
            loss.backward()
            optimizer.step()

            LOSS.append(loss.item())
            progress_bar.update(1)
        progress_bar.close()

        auc_test_, data_test = calculate_metrics(model, data_test_loader)

        if auc_test_ > auc_test:
            auc_test = auc_test_
            epoch_test = epoch
            early_stop = 0
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / f'checkpoint_max_test_auc.pth')
            data_test.to_csv(args.checkpoint_dir / f'temp_report_test.csv')
        print('Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(LOSS)))
        print(f'Epoch: {epoch} test AUC: {auc_test_}')
        print(f'Current best test AUC: {auc_test} at epoch {epoch_test}')

        scheduler.step()

    model = WISE_BRCA_combined(head_hidden_1=1536, head_hidden_2=768, head_hidden_3=384, dropout=0.40).cuda()
    ckpt = torch.load(
        f'{args.checkpoint_path}/WISE_BRCA_combined.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    auc_train_validation, data_train_validation = calculate_metrics(model, data_train_validation_loader)
    auc_test, data_test = calculate_metrics(model, data_test_loader)
    auc_YYH, data_YYH = calculate_metrics(model, data_YYH_loader)

    print(
        f'Epoch: train AUC: {auc_train_validation},test AUC: {auc_test}, YYH AUC: {auc_YYH}')
    data_train_validation.to_csv(args.checkpoint_dir / f'temp_report_train_validation.csv')
    data_test.to_csv(args.checkpoint_dir / f'temp_report_test.csv')
    data_YYH.to_csv(args.checkpoint_dir / f'temp_report_YYH.csv')
    print('Finish!')

if __name__ == '__main__':
    sys.stdout = open(f'./checkpoint/log.txt', 'w')
    train()