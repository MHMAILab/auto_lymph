import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import json
import logging
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from apex import amp
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from classification.data.image_producer import GridImageDataset  # noqa
from classification.model import MODELS  # noqa


cudnn.enabled = True
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--cfg_path', default='./classification/configs/resnet50_crf.json', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--ckpt_path_save', '-ckpt_s',
                    default='./classification/runs/', help='checkpoint path to save')
parser.add_argument('--log_path', '-lp',
                    default='./classification/log/', help='log path')
parser.add_argument('--num_workers', default=4, type=int,
                    help='number of workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1,2,3,4', type=str, help='comma separated indices of GPU to use,'
                                                                        ' e.g. 0,1 for using GPU_0'
                                                                        ' and GPU_1, default 0.')
parser.add_argument('--start_epoch', '-s', default=0,
                    type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=1000,
                    type=int, help='end epoch')
parser.add_argument('--experiment_id', '-eid',
                    default='0', help='experiment id')
parser.add_argument('--experiment_name', '-name',
                    default='resnet50_crf', help='experiment name')

use_cuda = True
args = parser.parse_args()
device = torch.device("cuda" if use_cuda else "cpu")

log_path = os.path.join(
    args.log_path, args.experiment_name + "_" + str(args.experiment_id))
print("log_path:", log_path)

ckpt_path_save = os.path.join(
    args.ckpt_path_save, args.experiment_name + "_" + str(args.experiment_id))
if not os.path.exists(ckpt_path_save):
    os.mkdir(ckpt_path_save)
print("ckpt_path_save:", ckpt_path_save)

loss_fn = nn.BCEWithLogitsLoss().to(device)


def load_checkpoint(args, net):
    print("Use ckpt: ", args.ckpt)
    assert len(args.ckpt) != 0, "Please input a valid ckpt_path"
    checkpoint = torch.load(args.ckpt)
    pretrained_dict = checkpoint['state_dict']
    net.load_state_dict(pretrained_dict)
    return net


def valid_epoch(summary, model, dataloader):
    model.eval()
    time_now = time.time()
    loss_sum = 0
    acc_sum = 0
    steps = len(dataloader)
    batch_size = dataloader.batch_size
    grid_size = dataloader.dataset._grid_size
    dataiter = iter(dataloader)
    TP = 0
    FP = 0
    FN = 0
    y = []
    scores = []
    with torch.no_grad():
        for step in range(steps):
            data, target, pth = next(dataiter)
            data = data.to(device)
            target = target.to(device)
            target = target.view(-1)
            output = model(data)
            output = output.view(-1)
            loss = loss_fn(output, target)

            output = output.sigmoid()
            predict = torch.zeros_like(target)
            predict[output > 0.5] = 1
            y.extend(list(target.cpu().numpy()))
            scores.extend(list(output.cpu().numpy()))
            TP += (predict[target == 1] ==
                   1).type(torch.cuda.FloatTensor).sum().data.item()
            FP += (predict[target == 0] ==
                   1).type(torch.cuda.FloatTensor).sum().data.item()
            FN += (predict[target == 1] ==
                   0).type(torch.cuda.FloatTensor).sum().data.item()
            acc = (predict == target).type(torch.cuda.FloatTensor).sum().item()
            acc_data = acc * 1.0 / (batch_size * grid_size)
            # if acc_data <= 0.95:
            #     record_txt.writelines([pth[0], ',', str(acc_data), '\n'])
            loss_data = loss.item()
            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, Step : {}, Testing Loss : {:.5f}, '
                'Testing Acc : {:.3f},Run Time : {:.2f}'
                .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    step, loss_data, acc_data, time_spent))
            summary['step'] += 1
            loss_sum += loss_data
            acc_sum += acc
        y = np.array(y)
        scores = np.array(scores)
        fpr, tpr, thresholds = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=0.9, color='slateblue',
                 label='ROC curve (AUC = %0.3f)' % roc_auc)
        plt.xlabel('Sensitivity')
        plt.ylabel('Secificity')
        plt.plot([0, 1], [0, 1], '--', color='lightskyblue')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.grid(linestyle='-.')
        plt.legend()
        plt.savefig('./roc_cls.png')
        plt.savefig('./roc_cls.eps', format='eps', dpi=1000)

        # Precision = TP / (TP + FP)
        # Recall = TP / (TP + FN)
        # F1 = 2 * Precision * Recall / (Precision + Recall)

        # summary['Precision'] = Precision
        # summary['Recall'] = Recall
        # summary['F1'] = F1
        summary['loss'] = loss_sum / steps
        summary['acc'] = acc_sum / (steps * batch_size * grid_size)
        print('acc:', summary['acc'])
    return summary


def run():
    with open(args.cfg_path) as f:
        cfg = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_valid = cfg['test_batch_size'] * num_GPU
    num_workers = args.num_workers * num_GPU
    model = torch.jit.load('./model/classification.pt')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=1e-4)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = DataParallel(model, device_ids=None)
    epoch = 0
    summary_valid = {'loss': float(
        'inf'), 'step': 0, 'acc': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}
    dataset_valid = GridImageDataset(cfg['data_path'],
                                     cfg['label_path'],
                                     cfg['image_size'],
                                     way="valid",
                                     epoch=epoch)
    print(len(dataset_valid))
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=batch_size_valid,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)
    summary_valid = valid_epoch(summary_valid, model, dataloader_valid)


def main():

    logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    # record_txt = open('record.txt', 'w+')
    main()
    # record_txt.close()
