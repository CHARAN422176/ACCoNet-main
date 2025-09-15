import torch
import torch.nn.functional as F
import imageio
import numpy as np
import pdb, os, argparse
import time

from model.ACCoNet_VGG_models import ACCoNet_VGG
from model.ACCoNet_Res_models import ACCoNet_Res
from data import test_dataset

torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

# --------------------- Metrics ---------------------
def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def f_measure(pred, gt, beta=0.3):
    pred_bin = (pred > 0.5).astype(np.float32)
    tp = np.sum(pred_bin * gt)
    prec = tp / (np.sum(pred_bin) + 1e-8)
    rec = tp / (np.sum(gt) + 1e-8)
    f = (1 + beta) * prec * rec / (beta * prec + rec + 1e-8)
    return f

def s_measure(pred, gt):
    alpha = 0.5
    y = np.mean(gt)
    if y == 0:
        return 1 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)
    else:
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        o_fg = np.sum(fg) / (np.sum(gt) + 1e-8)
        o_bg = np.sum(bg) / (np.sum(1 - gt) + 1e-8)
        s = alpha * o_fg + (1 - alpha) * o_bg
        return s

def e_measure(pred, gt):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    fm = np.mean(pred)
    gt_mean = np.mean(gt)
    align_matrix = 2 * (pred - fm) * (gt - gt_mean) / (
        (pred - fm) ** 2 + (gt - gt_mean) ** 2 + 1e-8
    )
    return np.mean((align_matrix + 1) ** 2 / 4)

# --------------------- Model ---------------------
if opt.is_ResNet:
    model = ACCoNet_Res()
    model.load_state_dict(torch.load('./models/ACCoNet_ResNet/ACCoNet_Res.pth.39'))
else:
    model = ACCoNet_VGG()
    model.load_state_dict(torch.load('/kaggle/input/acconet/pytorch/default/2/EORSSD_ACCoNet_VGG.pth.39'))

model.cuda()
model.eval()

# --------------------- Dataset ---------------------
test_datasets = ['EORSSD']
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

    # initialize metric sums
    mae_sum = 0
    f_sum = 0
    s_sum = 0
    e_sum = 0
    time_sum = 0

    print(f"Testing on {dataset}...")

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        time_start = time.time()
        res, s2, s3, s4, s5, s1_sig, s2_sig, s3_sig, s4_sig, s5_sig = model(image)
        time_end = time.time()
        time_sum += (time_end - time_start)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # save result
        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))

        # compute metrics
        mae_sum += mae(res, gt)
        f_sum += f_measure(res, gt)
        s_sum += s_measure(res, gt)
        e_sum += e_measure(res, gt)

    # average metrics
    mae_avg = mae_sum / test_loader.size
    f_avg = f_sum / test_loader.size
    s_avg = s_sum / test_loader.size
    e_avg = e_sum / test_loader.size

    print(f"Running time per image: {time_sum/test_loader.size:.5f} sec")
    print(f"Average FPS: {test_loader.size/time_sum:.4f}")
    print(f"MAE: {mae_avg:.4f}, F-measure: {f_avg:.4f}, S-measure: {s_avg:.4f}, E-measure: {e_avg:.4f}")
