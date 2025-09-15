import torch
import torch.nn.functional as F
import imageio
import numpy as np
import pdb, os, argparse
from scipy import misc
import time

from model.ACCoNet_VGG_models import ACCoNet_VGG
from model.ACCoNet_Res_models import ACCoNet_Res
from data import test_dataset

# Evaluation metric functions
from skimage.metrics import structural_similarity as ssim  # for S-measure approximation
from skimage.metrics import mean_squared_error  # for MAE

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def f_measure(pred, gt, beta=0.3):
    pred_bin = (pred >= 0.5).astype(np.float32)
    tp = (pred_bin * gt).sum()
    precision = tp / (pred_bin.sum() + 1e-8)
    recall = tp / (gt.sum() + 1e-8)
    f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    return f

def s_measure(pred, gt):
    # Using a simple approximation with SSIM
    pred = (pred*255).astype(np.uint8)
    gt = (gt*255).astype(np.uint8)
    s = ssim(pred, gt, data_range=255)
    return s

def e_measure(pred, gt):
    # Enhanced alignment measure approximation
    pred_bin = (pred >= 0.5).astype(np.float32)
    gt_bin = (gt >= 0.5).astype(np.float32)
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    e = (2 * inter + 1e-8) / (union + inter + 1e-8)
    return e

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path = './dataset/test_dataset/'

if opt.is_ResNet:
    model = ACCoNet_Res()
    model.load_state_dict(torch.load('./models/ACCoNet_ResNet/ACCoNet_Res.pth.39'))
else:
    model = ACCoNet_VGG()
    model.load_state_dict(torch.load('/kaggle/input/acconet/pytorch/default/2/EORSSD_ACCoNet_VGG.pth.39'))

model.cuda()
model.eval()

test_datasets = ['EORSSD']
# test_datasets = ['ORSSD']

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = './results/ResNet50/' + dataset + '/'
    else:
        save_path = './results/VGG/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = '/kaggle/input/eorssd/test-images/'
    gt_root = '/kaggle/input/eorssd/test-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    time_sum = 0
    f_sum = 0
    s_sum = 0
    e_sum = 0
    mae_sum = 0

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(image)
        time_end = time.time()
        time_sum += (time_end-time_start)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # Save prediction
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))

        # Metrics
        f_sum += f_measure(res, gt)
        s_sum += s_measure(res, gt)
        e_sum += e_measure(res, gt)
        mae_sum += mae(res, gt)

        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))
            print('F-measure: {:.4f}'.format(f_sum/test_loader.size))
            print('S-measure: {:.4f}'.format(s_sum/test_loader.size))
            print('E-measure: {:.4f}'.format(e_sum/test_loader.size))
            print('MAE: {:.4f}'.format(mae_sum/test_loader.size))
