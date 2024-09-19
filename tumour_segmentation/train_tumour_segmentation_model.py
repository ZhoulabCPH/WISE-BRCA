import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import argparse
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, accuracy_score
from model import Tumour_segmentation
from datasets import BC, Transform, Transform_
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from sklearn.model_selection import KFold


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False
seed = 0
setup_seed(seed)
def get_args(patch_size):
    parser = argparse.ArgumentParser(description='Breast cancer tumor region segmentation')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--patients_train_validation', default=rf'../datasets/clinical_data/BCSS/patients_train_validation_{patch_size}.csv')
    parser.add_argument('--patches_train_validation', default=rf'../datasets/clinical_data/BCSS/codebook_train_validation_{patch_size}.csv')
    parser.add_argument('--patches_test', default=rf'../datasets/clinical_data/BCSS/codebook_test_{patch_size}.csv')
    parser.add_argument('--patches_dir', default=rf'../datasets/BCSS_patches_{patch_size}/')
    parser.add_argument('--checkpoint-dir',
                        default=rf'./checkpoint/Tumour_segmentation_{patch_size}/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('--wd', default=7e-4)
    parser.add_argument('--T_max', default=5)
    args = parser.parse_args()
    return args


def calculate_metrics(model,data_loader):
    softmax = nn.Softmax(dim=1)
    data = pd.DataFrame()
    model.eval()

    score = []
    label = []
    pre_label = []
    for step, (patches, batch_labels) in enumerate(data_loader):
        patches = patches.cuda()
        pred_ = model.forward(patches)
        pred_ = softmax(pred_)
        score.append(pred_.detach().cpu().numpy()[:, 1])
        pre_label = pre_label + list(pred_.argmax(dim=1).detach().cpu().numpy())
        label = label + list(batch_labels.detach().cpu().numpy())
    score = np.concatenate(score)
    auc = roc_auc_score(label, list(score), multi_class='ovo')
    acc = accuracy_score(label, pre_label)
    data['label'] = list(label)
    data['score'] = list(score)

    return auc,acc, data


def train(patch_size):
    kf = KFold(n_splits=5, shuffle=True)
    args = get_args(patch_size)
    patients = pd.read_csv(args.patients_train_validation)
    patches_train_validation = pd.read_csv(args.patches_train_validation)
    l_cl = nn.CrossEntropyLoss()

    for fold, (train_index, validation_index) in enumerate(kf.split(list(patients.iloc[:, 0]))):
        early_stop_epoch = 0
        print(f'Fold {fold} training beginning!')
        patients_train = patients.iloc[train_index]
        patients_validation = patients.iloc[validation_index]
        train_patches_index = []
        validation_patches_index = []
        for i in range(len(patches_train_validation)):
            patch = patches_train_validation.iloc[i]['patches']
            if patch.split('_')[0] in list(patients_train.iloc[:,0]):
                train_patches_index.append(i)
            if patch.split('_')[0] in list(patients_validation.iloc[:,0]):
                validation_patches_index.append(i)
        workspace_train = patches_train_validation.iloc[train_patches_index]
        workspace_validation = patches_train_validation.iloc[validation_patches_index]
        workspace_test = pd.read_csv(args.patches_test)
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = Tumour_segmentation().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.T_max, eta_min=1e-6)
        data_train = BC(workspace_train, args.patches_dir, Transform())
        data_train_loader = DataLoader(data_train, args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        data_validation = BC(workspace_validation, args.patches_dir, Transform_())
        data_validation_loader = DataLoader(data_validation, args.batch_size, shuffle=True, num_workers=2,drop_last=False)
        data_test = BC(workspace_test, args.patches_dir, Transform_())
        data_test_loader = DataLoader(data_test, args.batch_size, shuffle=True, num_workers=2,drop_last=False,)
        scaler = torch.cuda.amp.GradScaler()

        auc_train = 0
        auc_validation = 0
        epoch_train = 0
        epoch_validation = 0
        for epoch in range(args.epochs):
            LOSS = []
            model.train()
            for step, (patches, batch_labels) in enumerate(data_train_loader, start=epoch * len(data_train_loader)):
                patches = patches.cuda()
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=False):
                    pred = model.forward(patches)
                    loss = l_cl(pred.to(torch.float64), batch_labels.cuda().long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                LOSS.append(loss.item())


            if epoch % 5 == 0:
                auc_train_, acc_train, data_train = calculate_metrics(model, data_train_loader)
                if auc_train_ > auc_train:
                    auc_train = auc_train_
                    epoch_train = epoch
                    data_train.to_csv(args.checkpoint_dir / f'temp_report_train_{fold}.csv')
                print(f'Fold {fold} Epoch: {epoch} train AUC: {auc_train_}, accuracy {acc_train}')
                print(f'Fold {fold} Current best train AUC: {auc_train} at epoch {epoch_train}')

            auc_validation_, acc_validation,  data_validation = calculate_metrics(model, data_validation_loader)
            if auc_validation_ <= auc_validation:
                early_stop_epoch = early_stop_epoch + 1
            if auc_validation_ > auc_validation:
                early_stop_epoch = 0
                auc_validation = auc_validation_
                epoch_validation = epoch
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                             optimizer=optimizer.state_dict())
                torch.save(state, args.checkpoint_dir / f'checkpoint_max_validation_auc_{fold}.pth')
                data_validation.to_csv(args.checkpoint_dir / f'temp_report_validation_{fold}.csv')

            if early_stop_epoch >= 20:
                print('Early stop!')
                break
            print('Fold ' + str(fold) + 'Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(LOSS)))
            print(f'Fold {fold} Epoch: {epoch} validation AUC: {auc_validation_}, accuracy {acc_validation}')
            print(f'Fold {fold} Current best validation AUC: {auc_validation} at epoch {epoch_validation}')
            scheduler.step()

        model = Tumour_segmentation().cuda()
        ckpt = torch.load(
            rf'{args.checkpoint_dir}/checkpoint_max_validation_auc_{fold}.pth',
            map_location='cuda:0')
        model.load_state_dict(ckpt['model'])
        auc_test, acc_test,  data_test = calculate_metrics(model, data_test_loader)

        print(f'Fold {fold}  test AUC: {auc_test}, accuracy {acc_test}')
        data_test.to_csv(args.checkpoint_dir / f'temp_report_test_{fold}.csv')

        print('Finish!')
if __name__ == '__main__':
    patch_size = 224
    train(patch_size)