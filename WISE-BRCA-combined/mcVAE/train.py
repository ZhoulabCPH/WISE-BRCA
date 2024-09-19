import torch
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import numpy as np
import argparse
import random
import torch.backends.cudnn as cudnn

from model_mcVAE import mcVAE
from datasets import BC_multi_modal, collate_multi_modal
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
    parser = argparse.ArgumentParser(description='Multi-channel variational autoencoder')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size')

    parser.add_argument('--workspace_CHCAMS_train_validation', default=f'../../datasets/clinical_data/CHCAMS_data_train_validation.csv')
    parser.add_argument('--workspace_CHCAMS_test', default=f'../../datasets/clinical_data/CHCAMS_data_test.csv')
    parser.add_argument('--workspace_YYH', default=f'../../datasets/clinical_data/YYH_data.csv')
    parser.add_argument('--CHCAMS_slides_path_224',
                        default=r'../../datasets/WSIs_CTransPath_cluster_sample_224_CHCAMS')
    parser.add_argument('--CHCAMS_slides_path_512',
                        default=r'../../datasets/WSIs_CTransPath_cluster_sample_512_CHCAMS')
    parser.add_argument('--YYH_slides_path_224',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_224_YYH')
    parser.add_argument('--YYH_slides_path_512',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_512_YYH')
    parser.add_argument('--checkpoint-path',
                        default=f'./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('--wd', default=5e-4)
    parser.add_argument('--T_max', default=10)
    args = parser.parse_args()
    return args

def calculate_metrics(model,data_loader):
    with torch.no_grad():
        LOSS = []
        for step, slide in enumerate(data_loader):
            patches_224, patches_name_224, patches_512, patches_name_512, phenotype_feature, batch_labels, id_ = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['phenotype_feature'], slide['labels'], slide['id']
            l_mcVAE = model.forward(patches_224.cuda(), patches_512.cuda(), phenotype_feature.cuda())

            LOSS.append(l_mcVAE.item())


    return np.mean(LOSS)


def train():
    args = get_args()
    workspace_CHCAMS_train_validation = pd.read_csv(args.workspace_CHCAMS_train_validation)
    workspace_CHCAMS_test = pd.read_csv(args.workspace_CHCAMS_test)
    workspace_YYH = pd.read_csv(args.workspace_YYH)


    seed = 11
    setup_seed(seed)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = mcVAE().cuda()
    model.load_state_dict_from_checkpoint()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=50)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=50, after_scheduler=scheduler)
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
    min_loss = 999999
    for epoch in range(args.epochs):
        early_stop = early_stop + 1
        if early_stop > 25:
            print('Early stop!')
            break
        LOSS = []
        model.train()
        progress_bar = tqdm(total=len(data_train_validation_loader), desc="mcVAE training")
        for step, slide in enumerate(data_train_validation_loader, start=epoch * len(data_train_validation_loader)):
            patches_224, patches_name_224, patches_512, patches_name_512, phenotype_feature, batch_labels, id = slide[
                'patches_features_224'], slide['patches_names_224'], slide['patches_features_512'], slide[
                'patches_names_512'], slide['phenotype_feature'], slide['labels'], slide['id']

            optimizer.zero_grad()

            l_mcVAE = model.forward(patches_224.cuda(), patches_512.cuda(), phenotype_feature.cuda())
            loss = l_mcVAE
            loss.backward()
            optimizer.step()

            LOSS.append(loss.item())
            progress_bar.update(1)
        progress_bar.close()

        if np.mean(LOSS) < min_loss:
            min_loss = np.mean(LOSS)
            early_stop = 0
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / f'mcVAE.pth')

        print('Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(LOSS)))
        scheduler.step()

    model = mcVAE().cuda()
    ckpt = torch.load(
        f'{args.checkpoint_path}/mcVAE.pth',
        map_location='cuda:0')
    model.load_state_dict(ckpt['model'])
    CHCAMS_test_loss = calculate_metrics(model, data_test_loader)
    YYH_loss = calculate_metrics(model, data_YYH_loader)
    print(rf'CHCAMS test loss: {CHCAMS_test_loss}, YYH loss: {YYH_loss}')
    print('Finish!')

if __name__ == '__main__':

    sys.stdout = open(f'./checkpoint/log.txt', 'w')
    train()