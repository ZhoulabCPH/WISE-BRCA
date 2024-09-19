import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import pandas as pd
import numpy as np
import argparse
import random
import torch.backends.cudnn as cudnn


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from model import WISE_BRCA
from datasets import BC, collate
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

    parser = argparse.ArgumentParser(description='BRCA1/2 mutation prediction')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--patients', default=f'../datasets/clinical_data/WISE-BRCA/patients_train_validation.csv')
    parser.add_argument('--workspace_train_validation', default=f'../datasets/clinical_data/WISE-BRCA/workspace_train_validation.csv')
    parser.add_argument('--workspace_test', default=f'../datasets/clinical_data/WISE-BRCA/workspace_test.csv')
    parser.add_argument('--workspace_YYH', default=f'../datasets/clinical_data/WISE-BRCA/workspace_YYH.csv')
    parser.add_argument('--workspace_HMUCH', default=f'../datasets/clinical_data/WISE-BRCA/workspace_HMUCH.csv')
    parser.add_argument('--CHCAMS_slides_path_224',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_224_CHCAMS')
    parser.add_argument('--CHCAMS_slides_path_512',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_512_CHCAMS')
    parser.add_argument('--YYH_slides_path_224',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_224_YYH')
    parser.add_argument('--YYH_slides_path_512',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_512_YYH')
    parser.add_argument('--HMUCH_slides_path_224',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_224_HMUCH')
    parser.add_argument('--HMUCH_slides_path_512',
                        default=r'../datasets/WSIs_CTransPath_cluster_sample_512_HMUCH')
    parser.add_argument('--checkpoint-path',
                        default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=1e-5)
    parser.add_argument('--wd', default=5e-4)
    parser.add_argument('--T_max', default=10)
    args = parser.parse_args()
    return args


def calculate_metrics(model, data_loader):
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
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    args = get_args()

    slides_train_validation = pd.read_csv(args.workspace_train_validation)
    l_bce = nn.BCEWithLogitsLoss()

    labels = slides_train_validation.loc[:, 'BRCA_mut'].to_list()
    seed = 11
    setup_seed(seed)

    for fold, (train_index, validation_index) in enumerate(skf.split(list(slides_train_validation.iloc[:, 0]), labels)):
        print(f'Fold {fold} training beginning!')
        seed = 8+fold
        setup_seed(seed)

        workspace_CHCAMS_train = slides_train_validation.iloc[train_index]
        workspace_CHCAMS_validation = slides_train_validation.iloc[validation_index]
        workspace_CHCAMS_test = pd.read_csv(args.workspace_test)


        workspace_YYH = pd.read_csv(args.workspace_YYH)
        workspace_HMUCH = pd.DataFrame(args.workspace_YYH)

        args.checkpoint_path.mkdir(parents=True, exist_ok=True)
        model = WISE_BRCA().cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=50)
        data_train = BC(workspace_CHCAMS_train, f'{args.slides_path_224}', f'{args.slides_path_512}')
        data_train_loader = DataLoader(data_train, args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                                       collate_fn=collate)
        data_validation = BC(workspace_CHCAMS_validation, f'{args.slides_path_224}', f'{args.slides_path_512}')
        data_validation_loader = DataLoader(data_validation, args.batch_size, shuffle=True, num_workers=4,
                                            drop_last=False, collate_fn=collate)
        data_test = BC(workspace_CHCAMS_test,  f'{args.slides_dir_224}{args.aug}', f'{args.slides_dir_512}{args.aug}')
        data_test_loader = DataLoader(data_test, args.batch_size, shuffle=True, num_workers=4,
                                      drop_last=False, collate_fn=collate)


        data_YYH = BC(workspace_YYH, args.YYH_slides_path_224, args.YYH_slides_path_512)
        data_YYH_loader = DataLoader(data_YYH, args.batch_size, shuffle=True, num_workers=4,
                                         drop_last=False, collate_fn=collate)


        data_HMUCH = BC(workspace_HMUCH, args.HMUCH_224_path, args.HMUCH_512_path)
        data_HMUCH_loader = DataLoader(data_HMUCH, args.batch_size, shuffle=True, num_workers=4,
                                     drop_last=False, collate_fn=collate)

        early_stop = 0
        auc_validation = 0
        epoch_validation = 0
        for epoch in range(args.epochs):
            early_stop = early_stop + 1
            if early_stop > 2:
                print('Early stop!')
                break
            LOSS = []
            model.train()
            progress_bar = tqdm(total=len(data_train_loader), desc="MossMIL training")
            for step, slide in enumerate(data_train_loader, start=epoch * len(data_train_loader)):
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
            auc_train_, data_train = calculate_metrics(model, data_train_loader)
            auc_validation_, data_validation = calculate_metrics(model, data_validation_loader)
            if auc_validation_ > auc_validation:
                auc_validation = auc_validation_
                epoch_validation = epoch
                early_stop = 0
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_max_validation_auc_{fold}.pth')
                data_validation.to_csv(args.checkpoint_dir / f'temp_report_validation_{fold}.csv')
            print('Fold ' + str(fold) + 'Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(LOSS)))
            print(f'Fold {fold} Epoch: {epoch} train AUC: {auc_train_} validation AUC: {auc_validation_}')
            print(f'Fold {fold} Current best validation AUC: {auc_validation} at epoch {epoch_validation}')

            scheduler.step()

        model = WISE_BRCA().cuda()
        ckpt = torch.load(
            rf'{args.checkpoint_path}/WISE_BRCA_{fold}.pth',
            map_location='cuda:0')
        model.load_state_dict(ckpt['model'])
        auc_train, data_train = calculate_metrics(model, data_train_loader)
        auc_validation, data_validation = calculate_metrics(model, data_validation_loader)
        auc_test, data_test = calculate_metrics(model, data_test_loader)
        auc_YYH, data_YYH = calculate_metrics(model, data_YYH_loader)
        auc_HMUCH, data_HMUCH = calculate_metrics(model, data_HMUCH_loader)


        print(
            f'Fold {fold} train AUC: {auc_train}, validation AUC: {auc_validation}, test AUC: {auc_test}, YYH AUC: {auc_YYH},'
            f'HMUCH AUC: {auc_HMUCH}')
        data_train.to_csv(args.checkpoint_dir / f'temp_report_train_{fold}.csv')
        data_validation.to_csv(args.checkpoint_dir / f'temp_report_validation_{fold}.csv')
        data_test.to_csv(args.checkpoint_dir / f'temp_report_test_{fold}.csv')
        data_YYH.to_csv(args.checkpoint_dir / f'temp_report_YYH_{fold}.csv')
        data_HMUCH.to_csv(args.checkpoint_dir / f'temp_report_HMUCH_{fold}.csv')

    print('Finish!')

if __name__ == '__main__':

    sys.stdout = open('./checkpoint/log.txt', 'w')
    train()