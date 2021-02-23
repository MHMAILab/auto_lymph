from tensorboardX import SummaryWriter
from option import Options
from image_producer import ImageDataset
from segmentation.models.UNET.UNet import UNet
from segmentation import utils
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable


class Trainer():
    def __init__(self, args):
        self.args = args
        log_path = 'segmentation/runs/log/' + self.args.model
        if not os.path.exists('segmentation/runs'):
            os.mkdir('segmentation/runs')
        if not os.path.exists('segmentation/runs/log'):
            os.mkdir('segmentation/runs/log')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        log_path = log_path + '/' + self.args.checkname
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        self.summary_writer = SummaryWriter(log_path)
        # dataset
        trainset = ImageDataset(
            args.data_path, args.mask_path, model=args.model)
        testset = ImageDataset(
            args.data_path, args.mask_path, model=args.model, way="valid")
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)

        self.nclass = trainset.num_class
        # model
        model = UNet(n_channels=1, n_classes=1)
        # optimizer using different LR
        params_list = [{'params': model.parameters(), 'lr': args.lr}, ]
        optimizer = torch.optim.SGD(params_list,
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # criterions
        self.BCEloss = nn.BCEWithLogitsLoss()
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.BCEloss = self.BCEloss.cuda()
            self.model = torch.nn.DataParallel(self.model)
        # resuming checkpoint
        if args.resume:
            checkpoint = torch.load(args.ckpt)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))
        self.best_pred = 0.0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        train_recall = 0
        acc = 0
        for i, (image, target, _, _) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            image = image.cuda()
            target = target.cuda()
            outputs = self.model(image)
            outputs = outputs.squeeze()
            loss = self.BCEloss(outputs, target.float())
            predict = outputs > 0.5
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            size = predict.numel()
            acc += ((predict == target.byte()
                     ).sum().type(torch.cuda.FloatTensor).item() * 1.0 / size)
            TP = (predict[target == 1] == 1).sum().type(
                torch.cuda.FloatTensor).item()
            FN = (predict[target == 1] == 0).sum().type(
                torch.cuda.FloatTensor).item()
            Recall = TP / (TP + FN)
            train_recall += Recall
            tbar.set_description('Train loss: %.3f,Train ACC %.3f,Train recall %.3f' % (
                train_loss / (i + 1), acc / (i + 1), train_recall / (i + 1)))
        self.summary_writer.add_scalar(
            'train/loss', train_loss / len(self.trainloader), epoch)
        self.summary_writer.add_scalar(
            'train/recall', train_recall / len(self.trainloader), epoch)
        self.summary_writer.add_scalar(
            'train/acc', acc / len(self.trainloader), epoch)
        if epoch % 1 == 0:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)

    def validation(self, epoch):
        # Fast test during the training

        def eval_batch(model, image, target, img_name):
            outputs = model(image)
            outputs = outputs.squeeze()
            predict = outputs > 0.5
            if epoch % 10 == 0:
                for j in range(len(img_name)):
                    tp = (predict[j][target[j] == 1] == 1).sum().type(
                        torch.cuda.FloatTensor)
                    fn = (predict[j][target[j] == 1] == 0).sum().type(
                        torch.cuda.FloatTensor)
                    Rc = tp / (tp + fn)
            tumor_num = (target == 1).sum().item()
            TP = (predict[target == 1] == 1).sum().type(
                torch.cuda.FloatTensor).item()
            FN = (predict[target == 1] == 0).sum().type(
                torch.cuda.FloatTensor).item()
            Recall = TP / (TP + FN)
            correct, labeled = utils.batch_pix_accuracy(predict.data, target)
            inter, union = utils.batch_intersection_union(predict.data, target)
            return correct, labeled, inter, union, Recall, tumor_num

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_recall = 0, 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        Acc = 0
        mIOU = 0
        for i, (image, target, img_cp, img_name) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union, Recall, tumor_num = eval_batch(
                    self.model, image, target, img_name)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union, Recall, tumor_num = eval_batch(
                        self.model, image, target, img_name)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_recall += Recall
            recall = 1.0 * total_recall / (len(self.valloader))
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            Acc += pixAcc
            mIOU += mIoU
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f, Recall: %.3f, tumor num: %.3f' % (pixAcc, mIoU, recall, tumor_num))
        self.summary_writer.add_scalar(
            'test/acc', Acc / len(self.valloader), epoch)
        self.summary_writer.add_scalar(
            'test/mIOU', mIOU / len(self.valloader), epoch)
        self.summary_writer.add_scalar('test/recall', recall, epoch)
        new_pred = (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


if __name__ == "__main__":
    print('start train')
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation(epoch)
