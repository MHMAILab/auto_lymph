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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from classification.data.image_producer import GridImageDataset  # noqa
from classification.model import MODELS  # noqa

cudnn.enabled = True
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--cfg_path', default='./classification/configs/resnet50_crf.json', metavar='CFG_PATH',
                    type=str,
                    help='Path to the config file in json format')
parser.add_argument('--ckpt_path_save', '-ckpt_s', default='./classification/runs/', help='checkpoint path to save')
parser.add_argument('--log_path', '-lp', default='./classification/log/', help='log path')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for each data loader, default 2.')
parser.add_argument('--device_ids', default='0,1,2,3,4', type=str, help='comma separated indices of GPU to use,'
                                                                        ' e.g. 0,1 for using GPU_0'
                                                                        ' and GPU_1, default 0.')
parser.add_argument('--start_epoch', '-s', default=0, type=int, help='start epoch')
parser.add_argument('--end_epoch', '-e', default=1000, type=int, help='end epoch')
parser.add_argument('--ckpt', '-ckpt', default='', help='checkpoint path to save')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--experiment_id', '-eid', default='0', help='experiment id')
parser.add_argument('--experiment_name', '-name', default='resnet50_crf', help='experiment name')
parser.add_argument('--drop_group', '-drop_group', default='3,4', help='drop groups')
parser.add_argument('--drop_prob', '-drop_prob', default='0.1', type=float, help='drop prob')
use_cuda = True
args = parser.parse_args()
device = torch.device("cuda" if use_cuda else "cpu")
if not os.path.exists(args.log_path):
    os.mkdir(args.log_path)
if not os.path.exists(args.ckpt_path_save):
    os.mkdir(args.ckpt_path_save)
log_path = os.path.join(args.log_path, args.experiment_name + "_" + str(args.experiment_id))
print("log_path:", log_path)

ckpt_path_save = os.path.join(args.ckpt_path_save, args.experiment_name + "_" + str(args.experiment_id))
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


def train_epoch(epoch, summary, model, optimizer, dataloader_train):
    model.train()
    steps = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    grid_size = dataloader_train.dataset._grid_size
    time_now = time.time()
    loss_sum = 0
    acc_sum = 0
    summary['epoch'] = epoch
    TP = 0
    FP = 0
    FN = 0
    y = []
    scores= []
    print(steps)
    for idx, (data, target) in enumerate(dataloader_train):
        data = data.to(device)
        target = target.to(device)
        target = target.view(-1)
        output = model(data)
        output = output.view(-1)
        loss = loss_fn(output, target)
        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()


        optimizer.step()
        output = output.sigmoid()
        predict = torch.zeros_like(target)
        predict[output > 0.5] = 1
        TP += (predict[target == 1] == 1).type(torch.cuda.FloatTensor).sum().data.item()
        FP += (predict[target == 0] == 1).type(torch.cuda.FloatTensor).sum().data.item()
        FN += (predict[target == 1] == 0).type(torch.cuda.FloatTensor).sum().data.item()
        acc = (predict == target).type(torch.cuda.FloatTensor).sum().item()
        acc_data = acc * 1.0 / (batch_size * grid_size)
        loss_data = loss.item()

        time_spent = time.time() - time_now
        time_now = time.time()
        logging.info(
            '{}, Epoch : {}, Step : {}, Training Loss : {:.5f}, '
            'Training Acc : {:.3f},Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1,
                                                    idx + 1, loss_data, acc_data, time_spent))
        summary['step'] += 1
        loss_sum += loss_data
        acc_sum += acc
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    summary['Precision'] = Precision
    summary['Recall'] = Recall
    summary['F1'] = F1
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / (steps * batch_size * grid_size)
    summary['epoch'] += 1

    return summary


def valid_epoch(summary, summary_writer, epoch, model, dataloader):
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
    print(steps)

    with torch.no_grad():
        for step in range(steps):
            data, target = next(dataiter)
            print("target:", torch.unique(target))
            data = data.to(device)
            target = target.to(device)
            target = target.view(-1)
            output = model(data)
            output = output.view(-1)
            loss = loss_fn(output, target)

            output = output.sigmoid()
            predict = torch.zeros_like(target)
            predict[output > 0.5] = 1
            TP += (predict[target == 1] == 1).type(torch.cuda.FloatTensor).sum().data.item()
            FP += (predict[target == 0] == 1).type(torch.cuda.FloatTensor).sum().data.item()
            FN += (predict[target == 1] == 0).type(torch.cuda.FloatTensor).sum().data.item()
            acc = (predict == target).type(torch.cuda.FloatTensor).sum().item()
            acc_data = acc * 1.0 / (batch_size * grid_size)
            loss_data = loss.item()
            time_spent = time.time() - time_now
            time_now = time.time()
            logging.info(
                '{}, Epoch : {}, Step : {}, Testing Loss : {:.5f}, '
                'Testing Acc : {:.3f},Run Time : {:.2f}'
                    .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), epoch + 1,
                    step, loss_data, acc_data, time_spent))
            summary['step'] += 1
            loss_sum += loss_data
            acc_sum += acc
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 * Precision * Recall / (Precision + Recall)
        summary['Precision'] = Precision
        summary['Recall'] = Recall
        summary['F1'] = F1
        summary['loss'] = loss_sum / steps
        summary['acc'] = acc_sum / (steps * batch_size * grid_size)

    return summary


def adjust_learning_rate(optimizer, epoch, cfg):
    """decrease the learning rate at 200 and 300 epoch"""
    lr = cfg['lr']
    if epoch >= 25:
        lr /= 2
    if epoch >= 50:
        lr /= 2
    if epoch >= 100:
        lr /= 2
    if epoch >= 150:
        lr /= 2
    if epoch >= 200:
        lr /= 2
    if epoch >= 225:
        lr /= 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def run():

    with open(args.cfg_path) as f:
        cfg = json.load(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    num_GPU = len(args.device_ids.split(','))
    batch_size_train = cfg['train_batch_size'] * num_GPU
    batch_size_valid = cfg['test_batch_size'] * num_GPU
    num_workers = args.num_workers * num_GPU
    # block setting
    drop_prob = [0.] * 4
    if args.drop_group:
        drop_probs = args.drop_prob
        drop_group = [int(x) for x in args.drop_group.split(',')]
        for block_group in drop_group:
            if block_group < 1 or block_group > 4:
                raise ValueError(
                    'drop_group should be a comma separated list of integers'
                    'between 1 and 4(drop_group:{}).'.format(args.drop_group)
                )
            drop_prob[block_group - 1] = drop_probs / 4.0 ** (4 - block_group)
    model = MODELS[cfg['model']](drop_prob=drop_prob)
    if args.resume:
        model = load_checkpoint(args, model)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                                weight_decay=1e-4)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model = DataParallel(model, device_ids=None)
    summary_train = {'epoch': 0, 'step': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'acc': 0}
    summary_valid = {'loss': float('inf'), 'step': 0, 'acc': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'epoch': 0}
    summary_writer = SummaryWriter(log_path)
    loss_valid_best = float('inf')
    lr = cfg['lr']
    for epoch in range(args.start_epoch, args.end_epoch):
        lr = adjust_learning_rate(optimizer, epoch, cfg)
        dataset_train = GridImageDataset(cfg['data_path'],
                                         cfg['label_path'],
                                         cfg['image_size'],
                                         epoch=epoch)
        print(len(dataset_train))
        dataloader_train = DataLoader(dataset_train,
                                      batch_size=batch_size_train,
                                      num_workers=num_workers,
                                      drop_last=True,
                                      shuffle=True)
        summary_train = train_epoch(epoch, summary_train, model,
                                    optimizer, dataloader_train)
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.module.state_dict()},
                   (ckpt_path_save + '/' + str(epoch) + '.ckpt'))
        summary_writer.add_scalar(
            'train/loss', summary_train['loss'], epoch)
        summary_writer.add_scalar(
            'train/acc', summary_train['acc'], epoch)
        summary_writer.add_scalar(
            'train/Precision', summary_train['Precision'], epoch
        )
        summary_writer.add_scalar(
            'train/Recall', summary_train['Recall'], epoch
        )
        summary_writer.add_scalar(
            'train/F1', summary_train['F1'], epoch
        )
        if epoch % 2 == 0:
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
            summary_valid = valid_epoch(summary_valid, summary_writer, epoch, model,
                                        dataloader_valid)
            summary_writer.add_scalar(
                'valid/loss', summary_valid['loss'], epoch)
            summary_writer.add_scalar(
                'valid/acc', summary_valid['acc'], epoch)
            summary_writer.add_scalar(
                'valid/Precision', summary_valid['Precision'], epoch
            )
            summary_writer.add_scalar(
                'valid/Recall', summary_valid['Recall'], epoch
            )
            summary_writer.add_scalar(
                'valid/F1', summary_valid['F1'], epoch
            )
        summary_writer.add_scalar('learning_rate', lr, epoch)
        if summary_valid['loss'] < loss_valid_best:
            loss_valid_best = summary_valid['loss']

            torch.save({'epoch': summary_train['epoch'],
                        'step': summary_train['step'],
                        'state_dict': model.module.state_dict()},
                       os.path.join(ckpt_path_save, 'best.ckpt'))

    summary_writer.close()


def main():
    logging.basicConfig(level=logging.INFO)

    run()


if __name__ == '__main__':
    main()
