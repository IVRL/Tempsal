import fnmatch
import os
import torch
import cv2

from tqdm import tqdm
from scipy.spatial import distance
from math import pi, sqrt, exp
from os.path import join
from torchvision import utils
from PIL import Image

import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

W = 640
H = 480
TIMESPAN = 5000
MAX_PIXEL_DISTANCE = 800
ESTIMATED_TIMESTAMP_WEIGHT = 0.006
RATIO = 0.9

FIXATION_PATH = '../data/fixations/'
FIX_VOL_PATH = '../data/fixation_volumes_'
SAL_VOL_PATH = '../data/saliency_volumes_'

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_colored_value(value, ref_value, increasing=True):
    coef = 1 if increasing else -1
    return (bcolors.FAIL if (coef * ref_value > coef * value) else bcolors.OKGREEN) + '{:.5f}'.format(value) + bcolors.ENDC

def get_filenames(path):
    return [file for file in sorted(os.listdir(path)) if fnmatch.fnmatch(file, 'COCO_*')]

def parse_fixations(filenames,
                    path_prefix,
                    etw=ESTIMATED_TIMESTAMP_WEIGHT, progress_bar=True):
    fixation_volumes = []
    filenames = tqdm(filenames) if progress_bar else filenames

    for filename in filenames:
        # 1. Extracting data from .mat files
        mat = sio.loadmat(path_prefix + filename + '.mat')
        gaze = mat["gaze"]

        locations = []
        timestamps = []
        fixations = []

        for i in range(len(gaze)):
            locations.append(mat["gaze"][i][0][0])
            timestamps.append(mat["gaze"][i][0][1])
            fixations.append(mat["gaze"][i][0][2])

        # 2. Matching fixations with timestamps
        fixation_volume = []
        for i, observer in enumerate(fixations):
            fix_timestamps = []
            fix_time = TIMESPAN / (len(observer) + 1)
            est_timestamp = fix_time

            for fixation in observer:
                distances = distance.cdist([fixation], locations[i], 'euclidean')[0][..., np.newaxis]
                time_diffs = abs(timestamps[i] - est_timestamp)
                min_idx = (etw * time_diffs + distances).argmin()

                fix_timestamps.append([min(timestamps[i][min_idx][0], TIMESPAN), fixation.tolist()])
                est_timestamp += fix_time

            if (len(observer) > 0):
                fixation_volume.append(fix_timestamps)

        fixation_volumes.append(fixation_volume)

    return fixation_volumes

def get_saliency_volume(fixation_volume, conv2D, time_slices):
    fixation_map = torch.cuda.FloatTensor(time_slices,H,W).fill_(0)

    for ts, coords in fixation_volume:
        for (x, y) in coords:
            fixation_map[ts,y-1,x-1] = 1

    saliency_volume = conv2D.forward(fixation_map)
    return saliency_volume / saliency_volume.max()

def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img,(k_size,k_size),0)
    return torch.FloatTensor(bl)

def visualize_model(model, loader, device, args):
    with torch.no_grad():
        model.eval()
        os.makedirs(args.results_dir, exist_ok=True)
        
        for (img, img_id, sz) in tqdm(loader):
            img = img.to(device)
            
            pred_map = model(img)
            if type(pred_map) is tuple:
                pred_map = pred_map[1]
            pred_map = pred_map.cpu().squeeze(0).numpy()
            pred_map = cv2.resize(pred_map, (sz[0], sz[1]))
            
            pred_map = torch.FloatTensor(blur(pred_map))
            img_save(pred_map, join(args.results_dir, img_id[0]), normalize=True)

def img_save(tensor, fp, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0, format=None):
    grid = utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    ''' Add 0.5 after unnormalizing to [0, 255] to round to nearest integer '''
    
    ndarr = torch.round(grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr[:,:,0])
    #fp = fp[:-4] + '.png'
    fp = '.png'
    print(fp)
    im.save("1.png", format=format, compress_level=0)

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.past = np.array([])
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
        self.past = np.append(self.past,val.cpu())
        
    def get_std (self):
        return np.std(self.past)
         
    def get_median (self):
        return np.median(self.past)


def im2heat(pred_dir, a, gt, exten='.png'):
    pred_nm = pred_dir + a + exten
    pred = cv2.imread(pred_nm, 0)
    heatmap_img = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
    heatmap_img = convert(heatmap_img)
    pred = np.stack((pred, pred, pred),2).astype('float32')
    pred = pred / 255.0
    
    return np.uint8(pred * heatmap_img + (1.0-pred) * gt)

def get_heat_image(image):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)

def format_image(heatmap, image, max_value):
    extended_map = heatmap / max_value
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    factors = np.clip(2 * extended_map, 0, 1)
    hsv[:,:,1] = np.uint8(factors * hsv[:,:,1])
    hsv[:,:,2] = np.uint8((RATIO * factors + (1 - RATIO)) * hsv[:,:,2])
    
    return get_heat_image(extended_map[:,:,np.newaxis]), cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def animate(gt_vol, pred_vol, image):
    fig = plt.figure(figsize=(16, 16))

    gt_max = np.max(gt_vol);
    pred_max = np.max(pred_vol);
    formatted_images = []
    
    for (gt_map, pred_map) in zip(gt_vol, pred_vol):
        gt_heatmap, gt_image_heatmap = format_image(gt_map, image, gt_max)
        gt_im = np.concatenate((gt_heatmap, gt_image_heatmap), 1)

        pred_heatmap, pred_image_heatmap = format_image(pred_map, image, pred_max)
        pred_im = np.concatenate((pred_heatmap, pred_image_heatmap), 1)
        
        diff = 0.5 + ((gt_map / gt_max) - (pred_map / pred_max)) / 2
        diff_im = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * diff[:,:,np.newaxis]), cv2.COLORMAP_TWILIGHT), cv2.COLOR_BGR2RGB)
        diff_im = np.concatenate((diff_im, image), 1)
        formatted_images.append([plt.imshow(np.concatenate((gt_im, pred_im, diff_im), 0), animated=True)])

    return animation.ArtistAnimation(fig, formatted_images, interval=500, blit=True, repeat_delay=1000)

def animate_single_heatmap(gt_vol, image):
    fig = plt.figure(figsize=(6.4, 4.8),frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    gt_max = np.max(gt_vol);
    formatted_images = []
    plt.axis('off')
    for gt_map in gt_vol:
        gt_heatmap, gt_image_heatmap = format_image(gt_map, image, gt_max)
        gt_im = gt_heatmap
        formatted_images.append([ax.imshow(gt_im, animated=True)])
    return animation.ArtistAnimation(fig, formatted_images, interval=1000, blit=True, repeat_delay=1000)


def gauss(n, sigma):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

class GaussianBlur1D(nn.Module):
    def __init__(self, time_slices):
        super(GaussianBlur1D, self).__init__()
        sigma = 2 * time_slices / 25
        self.size = 2 * int(4 * sigma + 0.5) + 1
        kernel = gauss(self.size, sigma)
        kernel = torch.cuda.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        pad = int(self.size/2)
        temp = F.conv1d(x, self.weight.view(1, 1, -1, 1, 1), padding=pad)
        return temp[:,:,:,pad:-pad,pad:-pad]

class GaussianBlur2D(nn.Module):
    def __init__(self):
        super(GaussianBlur2D, self).__init__()
        self.size = 201
        kernel = gauss(self.size, 25)
        kernel = torch.cuda.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        pad = int(self.size/2)
        temp = F.conv1d(x.unsqueeze(0).unsqueeze(0), self.weight.view(1, 1, 1, -1, 1), padding=pad)
        temp = temp[:,:,pad:-pad,:,pad:-pad]
        temp = F.conv1d(temp, self.weight.view(1, 1, 1, 1, -1), padding=pad)
        return temp[:,:,pad:-pad,pad:-pad]
