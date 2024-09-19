import torch
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import argparse
import random
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from model import WISE_BRCA
from datasets import BC, BC_, collate
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


def get_args():
    parser = argparse.ArgumentParser(description='BRCA mutation prediction')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='mini-batch size')

    parser.add_argument('--workspace_CHCAMS_pretrained',
                        default=f'../datasets/clinical_data/workspace_CHCAMS_pretrained_biopsy.csv')
    parser.add_argument('--workspace_CHCAMS_test',
                        default=f'../datasets/clinical_data/workspace_CHCAMS_test_biopsy.csv')
    parser.add_argument('--workspace_YYH', default=f'../datasets/clinical_data/workspace_YYH_biopsy.csv')
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
    parser.add_argument('--lr', default=1e-3)
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
            patches_224, patches_name_224, patches_512, patches_name_512, batch_labels, id_ = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['labels'], slide['id']
            id = id+id_
            pred_ = model.forward(patches_224.cuda(), patches_512.cuda())

            score = np.append(score, (sigmoid(pred_).detach().cpu().numpy()))

            label = label+list(batch_labels.detach().cpu().numpy())

        auc = roc_auc_score(label, score)
        data['id'] = list(id)
        data['label'] = list(label)
        data['score'] = list(score)

    return auc, data


def train():
    args = get_args()
    workspace_CHCAMS_pretrained = pd.read_csv(args.workspace_CHCAMS_pretrained)
    workspace_CHCAMS_test = pd.read_csv(args.workspace_CHCAMS_test)
    workspace_YYH = pd.read_csv(args.workspace_YYH)

    l_bce = nn.BCEWithLogitsLoss()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = WISE_BRCA().cuda()
    ckpt = torch.load(
        rf'../checkpoints/WISE_BRCA.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=20)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)
    data_pretrained = BC(workspace_CHCAMS_pretrained, f'{args.CHCAMS_slides_path_224}', f'{args.CHCAMS_slides_path_512}')
    data_pretrained_loader = DataLoader(data_pretrained, args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                                   collate_fn=collate)
    data_test = BC(workspace_CHCAMS_test, f'{args.CHCAMS_slides_path_224}', f'{args.CHCAMS_slides_path_512}')
    data_test_loader = DataLoader(data_test, args.batch_size, shuffle=True, num_workers=4,
                                        drop_last=False, collate_fn=collate)

    data_YYH = BC(workspace_YYH, args.YYH_slides_path_224, args.YYH_slides_path_512)
    data_YYH_loader = DataLoader(data_YYH, args.batch_size, shuffle=True, num_workers=4, drop_last=False,
                           collate_fn=collate)

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
        progress_bar = tqdm(total=len(data_pretrained_loader), desc="MossMIL training")
        for step, slide in enumerate(data_pretrained_loader, start=epoch * len(data_pretrained_loader)):
            patches_224, patches_name_224, patches_512, patches_name_512, batch_labels, id = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['labels'], slide['id']

            optimizer.zero_grad()

            pred = model.forward(patches_224.cuda(), patches_512.cuda())
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
            torch.save(state, args.checkpoint_dir / f'checkpoint_max_validation_auc.pth')
            data_test.to_csv(args.checkpoint_dir / f'temp_report_validation.csv')
        print('Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(LOSS)))
        print(f'Epoch: {epoch}, CHCAMS test AUC: {auc_test_}')
        print(f'Current best validation AUC: {epoch_test} at epoch {epoch_test}')
        scheduler.step()
    model = WISE_BRCA().cuda()
    ckpt = torch.load(
        f'../checkpoints/WISE_BRCA_biopsy.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    auc_pretrained, data_pretrained = calculate_metrics(model, data_pretrained_loader)
    auc_test, data_test = calculate_metrics(model, data_test_loader)
    auc_YYH, data_YYH = calculate_metrics(model, data_YYH_loader)

    print(
        f'Epoch: CHCAMS pretrained AUC: {auc_pretrained}, CHCAMS test AUC: {auc_test}, YYH AUC: {auc_YYH}')
    data_pretrained.to_csv(args.checkpoint_dir / f'temp_report_pretrained.csv')
    data_test.to_csv(args.checkpoint_dir / f'temp_report_test.csv')
    data_YYH.to_csv(args.checkpoint_dir / f'temp_report_YYH.csv')
    print('Finish!')
if __name__ == '__main__':
    sys.stdout = open('./checkpoints/log.txt', 'w')
    train()