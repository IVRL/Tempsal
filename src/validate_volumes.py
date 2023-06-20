import argparse
import os
import torch
import cv2

import numpy as np
import torch.nn as nn

from collections import OrderedDict
from dataloader import SaliconVolDataset
from tqdm import tqdm
from utils import *
from model import PNASVolModel#, VolModel
from loss import *
from matplotlib.image import imread

parser = argparse.ArgumentParser()

parser.add_argument('--model_val_path',default="../models/model.pt", type=str)
parser.add_argument('--model_path',default="../salicon_pnas.pt", type=str)
parser.add_argument('--model_vol_path',default="", type=str)

parser.add_argument('--time_slices',default=5, type=int)
parser.add_argument('--normalize',default=False, type=str)
parser.add_argument('--samples',default=25, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--dataset_dir',default="../data/", type=str)
parser.add_argument('--model',default="PNASVol", type=str)


def get_heat_image(image):
    return cv2.applyColorMap(np.uint8(255 * image), cv2.COLORMAP_HOT)


def format_image(heatmap, image, max_value):
    extended_map = heatmap / max_value
 
    return get_heat_image(extended_map), get_heat_image(extended_map)

def get_gt(imname):
    gt_dir = args.dataset_dir +   "maps/val/"
    try:
        img = cv2.imread(gt_dir+imname+".png",  cv2.IMREAD_GRAYSCALE)
    except:
        print("File not found : ", gt_dir+imname+".png")

    return img

def color_image_save(fin_pred,pnas_pred,image ):

        fig = plt.figure(figsize=(16, 16))
        preddname = val_pred_dir + val_img_ids[idx] + '_final_bw.png'
        print(idx)
        
        #cv2.imwrite(preddname,np.uint8(255 *pred_vol/np.max(pred_vol)))
        cv2.imwrite(preddname,np.uint8( fin_pred/np.max(fin_pred)*255))

        fin_max = np.max(fin_pred)
        pnas_max = np.max(pnas_pred)
        formatted_images = []

    #for (gt_map, pred_map) in zip(gt_vol, pred_vol):
        pnasname = val_pred_dir + val_img_ids[idx] + '_pnas_pred.png'
        finpreddname = val_pred_dir + val_img_ids[idx] + '_final_pred.png'

        pnas_heatmap, pnas_image_heatmap = format_image(pnas_pred, image, pnas_max)
      #  gt_im = np.concatenate((gt_heatmap, gt_image_heatmap), 1)
        cv2.imwrite(pnasname, pnas_heatmap)

        finpred_heatmap, finpred_image_heatmap = format_image(fin_pred, image, fin_max)
      #  pred_im = np.concatenate((pred_heatmap, pred_image_heatmap), 1)
        cv2.imwrite(finpreddname,finpred_heatmap)
        
        imgtname = val_pred_dir + val_img_ids[idx] + '_imgt.png'

        
        img_gt = get_gt( val_img_ids[idx])
     #   imgt_heatmap, imgt_heatmapwithimage = format_image(img_gt, image, pred_max)
        cv2.imwrite(imgtname,img_gt)
        
        
        
    #    formatted_images.append([plt.imshow(np.concatenate((gt_im, pred_im, diff_im), 0), animated=True)])
      #  plt.imsave(val_pred_dir + val_img_ids[idx] + '_pred.png')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == 'pnas':
    from model import PNASModel

    model =  PNASModel(load_weight=0)
if args.model == 'PNASVol':
    model = PNASVolModel(time_slices=args.time_slices)
elif args.model == 'Vol':
    model = VolModel(device, time_slices=args.time_slices)
elif args.model == "pnas_boosted_multiout":
    print("PNAS Boosted Model PNASBoostedModelMultilevel")
    from model import PNASBoostedModelMultilevelMultiOutput
    model = PNASBoostedModelMultilevelMultiOutput(device, args.model_path, args.model_vol_path, args.time_slices, train_model=False )
if args.model!="mobilenet":
    model = nn.DataParallel(model)

    
    
    
model = nn.DataParallel(model)
state_dict = torch.load(args.model_val_path)
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    #if 'module.module.pnas.module.module' not in k:
    k = 'module.module.' + k
#     else:
#         k = k.replace('features.module.', 'module.features.')
    new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict = False)
#model.load_state_dict(state_dict)

model = model.to(device)


data_dor = "5-original"
val_img_dir = "/sinergia/bahar/TemporalSaliencyPrediction/SimpleNet/images_for_poster/image/"
#val_img_dir = args.dataset_dir + "images/val/"
val_vol_dir = args.dataset_dir + "saliency_volumes_" + data_dor + "/val/"
val_fix_dir = args.dataset_dir + "fixation_volumes_" + data_dor + "/val/"
val_pred_dir = "/sinergia/bahar/TemporalSaliencyPrediction/SimpleNet/images_for_poster/prediction/"
#val_pred_dir = args.dataset_dir + "volumes/"

val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]
val_dataset = SaliconVolDataset(val_img_dir, val_vol_dir, val_fix_dir, val_img_ids, args.time_slices, selected_slices="")
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

with torch.no_grad():
    model.eval()
    os.makedirs(val_pred_dir, exist_ok=True)

    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for idx, (img, gt_vol, avg_vol) in enumerate(tqdm(val_loader)):
        img = img.to(device)
        gt_vol = gt_vol.to(device)
        if args.model == "pnas_boosted_multiout":
            fin_pred , pnas_pred, pred_vol = model(img)
            pnas_pred = pnas_pred.squeeze(0)
        else:
            pred_vol = model(img)
            fin_pred = pred_vol
            pnas_pred =  pred_vol
        pred_vol =   pred_vol.squeeze(0)
        gt_vol = gt_vol.squeeze(0)

        if args.normalize:
            avg_vol = avg_vol.to(device).squeeze(0)
            pred_vol = pred_vol * 2 + avg_vol - 1

        #pred_vol /= pred_vol.max()
        print("PRED VOL", pred_vol.shape)
        print("GT VOL",gt_vol.shape)
        print("fin_pred ",fin_pred.shape)
        print("pnas_pred ",pnas_pred.shape)
        cur_cc = cc(pred_vol, gt_vol)
        cur_kldiv = kldiv(pred_vol, gt_vol)
        cur_nss = nss(pred_vol, gt_vol)
        cur_sim = similarity(pred_vol, gt_vol)
        
        print(val_img_ids[idx] ," , CC " , cur_cc.item(), " , KL " ,cur_kldiv.item(), " , NSS " ,cur_nss.item()," , SIM " , cur_sim.item() )
        cc_loss.update(cur_cc)
        kldiv_loss.update(cur_kldiv)
        nss_loss.update(cur_nss)
        sim_loss.update(cur_sim)

        if idx < args.samples:
            pred_vol = np.swapaxes(pred_vol.detach().cpu().numpy(), 0, -1)
            pred_vol = np.swapaxes(cv2.resize(pred_vol, (H, W)), 0, -1)

            gt_vol = np.swapaxes(gt_vol.detach().cpu().numpy(), 0, -1)
            gt_vol = np.swapaxes(cv2.resize(gt_vol, (H, W)), 0, -1)

            img_path = os.path.join(val_img_dir, val_img_ids[idx] + '.jpg')
            img = imread(img_path)

            anim = animate_single_heatmap(gt_vol, img)
            anim.save(val_pred_dir + val_img_ids[idx] + '_gt.gif', writer=animation.PillowWriter(fps=1))
            
            anim = animate_single_heatmap(pred_vol, img)
            anim.save(val_pred_dir + val_img_ids[idx] + '_ours.gif', writer=animation.PillowWriter(fps=1))
            
            anim = animate(gt_vol, pred_vol, img)
            anim.save(val_pred_dir + val_img_ids[idx] + '.gif', writer=animation.PillowWriter(fps=2))
            
            fin_prednp = np.swapaxes(fin_pred.detach().cpu().numpy(), 0, -1)
            fin_prednp = np.swapaxes(cv2.resize(fin_prednp, (H, W)), 0, -1)
        
            pnas_prednp = np.swapaxes(pnas_pred.detach().cpu().numpy(), 0, -1)
            pnas_prednp = np.swapaxes(cv2.resize(pnas_prednp, (H, W)), 0, -1)

            color_image_save(fin_prednp,pnas_prednp ,img)
            plt.close('all')

        if idx > args.samples:
                break
            


    print('KLDIV : {:.5f}, CC : {:.5f}, SIM : {:.5f}, NSS : {:.5f}'.format(kldiv_loss.avg, cc_loss.avg, sim_loss.avg, nss_loss.avg))

        
        