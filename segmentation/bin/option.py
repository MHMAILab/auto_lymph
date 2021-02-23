import os
import argparse
import torch


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            segmentation')
        # model and dataset
        parser.add_argument('--model', type=str, default='unet',
                            help='model name (default: fcn)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--data_path', type=str,
                            default=['./dataset/seg/img/'],
                            help='data path')
        parser.add_argument('--mask_path', type=str,
                            default=['./dataset/seg/label/'],
                            help='mask path')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        # training hyper params
        parser.add_argument('--aux', action='store_true', default=False,
                            help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default=False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch_size', type=int, default=5,
                            metavar='N', help='input batch size for \
                            training')
        parser.add_argument('--test_batch_size', type=int, default=4,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr_scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight_decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--device_ids', default='4', type=str, help='comma separated indices of GPU to use,'
                            ' e.g. 0,1 for using GPU_0'
                            ' and GPU_1, default 0.')
        parser.add_argument('--no-cuda', action='store_true',
                            default=False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument(
            '--resume', '-r', action='store_true', help='resume from checkpoint')
        parser.add_argument('--ckpt', '-ckpt', default='0',
                            help='checkpoint path to load')
        parser.add_argument('--checkname', type=str, default='0',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default=False,
                            help='finetuning on a different dataset')
        parser.add_argument('--pre-class', type=int, default=1,
                            help='num of pre-trained classes \
                            (default: None)')
        # evaluation option

        parser.add_argument('--eval', action='store_true', default=False,
                            help='evaluating mIoU')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'pascal_voc': 50,
                'pascal_aug': 50,
                'pcontext': 80,
                'ade20k': 160,
            }
            args.epochs = epoches[args.dataset.lower()]
        args.batch_size = args.batch_size * len(args.device_ids.split(','))
        if args.lr is None:
            lrs = {
                'pascal_voc': 0.0001,
                'pascal_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.01,
            }
            args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size
        return args
