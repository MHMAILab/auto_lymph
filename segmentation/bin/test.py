# -*-coding:utf-8-*-
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from segmentation.bin.option import Options
import segmentation.utils as utils
from segmentation.bin.image_producer import ImageDataset
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
torch_ver = torch.__version__[:3]
args = Options().parse()
device = torch.device("cuda" if args.cuda else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
model = torch.jit.load('./model/segmentation.pt')
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=None)

kwargs = {'num_workers': 16, 'pin_memory': True}
testset = ImageDataset(args.data_path, args.mask_path,
                       model=args.model, way="valid")
valloader = data.DataLoader(testset, batch_size=1,
                            drop_last=False, shuffle=False, **kwargs)
print(len(testset))
with torch.no_grad():
    model.eval()
    total_inter, total_union = 0, 0
    tbar = tqdm(valloader, desc='\r')
    Acc = 0
    mIOU = 0
    y = None
    scores = None
    for i, (image, target, img_cp, img_name) in enumerate(tbar):
        image = image.to(device)
        target = target.to(device)
        # target = target.squeeze()

        print(target.shape)
        outputs = model(image)
        outputs = outputs.squeeze()
        predict = outputs > 0.5
        inter, union = utils.batch_intersection_union(outputs.data, target)
        total_inter += inter
        total_union += union
    IoU = np.float64(1.0) * total_inter / \
        (np.spacing(1, dtype=np.float64) + total_union)
    mIOU = IoU.mean()
    print('jaccard score:', mIOU)
