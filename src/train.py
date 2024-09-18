import argparse
import os
import torch
import sys
import time
import wandb
import torch.nn as nn
from tqdm import tqdm
from dataloader import SaliconDataset
from loss import *
from utils import AverageMeter
from utils import img_save
from torchvision import utils
import torch.nn.functional as nnf
from os.path import join
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=30, type=int)
parser.add_argument('--lr',default=1e-5, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--nss',default=False, type=bool)
parser.add_argument('--sim',default=False, type=bool)
parser.add_argument('--nss_emlnet',default=False, type=bool)
parser.add_argument('--nss_norm',default=False, type=bool)
parser.add_argument('--l1',default=False, type=bool)
parser.add_argument('--lr_sched',default=False, type=bool)
parser.add_argument('--dilation',default=False, type=bool)
parser.add_argument('--enc_model',default="pnas", type=str)
parser.add_argument('--optim',default="Adam", type=str)

parser.add_argument('--load_weight',default=1, type=int)
parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--step_size',default=5, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=-1.0, type=float)
parser.add_argument('--nss_emlnet_coeff',default=1.0, type=float)
parser.add_argument('--nss_norm_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--train_enc',default=1, type=int)

parser.add_argument('--dataset_dir',default="../data/", type=str)
parser.add_argument('--batch_size',default=32, type=int)
parser.add_argument('--log_interval',default=60, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--train_model',default=False, type=bool)
parser.add_argument('--time_slices',default=5, type=int)
parser.add_argument('--selected_slices',default="", type=str)
parser.add_argument('--results_dir',default="", type=str )

# Path to save the model weights
parser.add_argument('--model_val_path',default="model.pt", type=str)
# If the model type is pnas_boosted, specify the path of the pre-trained pnas model here
parser.add_argument('--model_path',default="", type=str)
# If the model type is pnas_boosted, specify the path of the pre-trained pnasvol model here
parser.add_argument('--model_vol_path',default="", type=str)


args = parser.parse_args()

train_img_dir = args.dataset_dir + "images/train/"
train_gt_dir = args.dataset_dir + "maps/train/"
train_fix_dir = args.dataset_dir + "fixation_maps/train/"

val_img_dir = args.dataset_dir + "images/val/"
val_gt_dir = args.dataset_dir + "maps/val/"
val_fix_dir = args.dataset_dir + "fixation_maps/val/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.enc_model == "pnas":
    print("PNAS Model")
    from model import PNASModel
    model = PNASModel(train_enc=bool(args.train_enc), load_weight=args.load_weight)

elif args.enc_model == "pnas_boosted_multi":
    print("PNAS Boosted Model PNASBoostedModelMultilevel")
    from model import PNASBoostedModelMultilevel
    model = PNASBoostedModelMultilevel(device, args.model_path, args.model_vol_path, args.time_slices, train_model=args.train_model,selected_slices = args.selected_slices )
    
    
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
model.to(device)


train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]
val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.no_workers)



def loss_func(pred_map, gt, fixations, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.nss:
        loss += args.nss_coeff * nss(pred_map, fixations)
    if args.l1:
        loss += args.l1_coeff * criterion(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)
    #print("Loss: ", loss)
    return loss

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    
    tic = time.time()

    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()
        pred_map, vol_pred = model(img)    

        assert pred_map.size() == gt.size()

        loss = loss_func(pred_map, gt, fixations, args)
        loss.backward()

        total_loss += loss.item()
        cur_loss += loss.item()

        optimizer.step()
        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
            wandb.log({"loss": cur_loss/args.log_interval})
            cur_loss = 0.0
            sys.stdout.flush()

    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)

def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()

    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        pred_map   , vol_pred = model(img)
     
        cc_loss.update(cc(pred_map, gt))
        kldiv_loss.update(kldiv(pred_map, gt))
        nss_loss.update(nss(pred_map, fixations))
        sim_loss.update(similarity(pred_map, gt))

    print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, (time.time()-tic)/60))
    wandb.log({"CC": cc_loss.avg, 'KLDIV': kldiv_loss.avg, 'NSS': nss_loss.avg, 'SIM': sim_loss.avg})
    sys.stdout.flush()

    return cc_loss.avg,cc_loss,kldiv_loss,nss_loss,sim_loss

params = list(filter(lambda p: p.requires_grad, model.parameters()))

if args.optim=="Adam":
    optimizer = torch.optim.Adam(params, lr=args.lr)
if args.optim=="Adagrad":
    optimizer = torch.optim.Adagrad(params, lr=args.lr)
if args.optim=="SGD":
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
if args.lr_sched:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

print(device)
best_loss = 0
for epoch in range(0, args.no_epochs):
    loss = train(model, optimizer, train_loader, epoch, device, args)

    with torch.no_grad():
                cc_loss,cc_loss_obj,kldiv_loss,nss_loss,sim_loss = validate(model, val_loader, epoch, device, args)
                cc_loss -=kldiv_loss.avg
                if epoch == 0 :
                    best_loss = cc_loss
                if best_loss <= cc_loss:
                    best_loss = cc_loss
                    print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
                    wandb.log({"Best/CC mean": cc_loss,"Best/CC median": cc_loss_obj.get_median(), "Best/CC std": cc_loss_obj.get_std(),
        "Best/KLD mean": kldiv_loss.avg,"Best/KLD median": kldiv_loss.get_median(), "Best/KLD std": kldiv_loss.get_std(),
        "Best/NSS mean": nss_loss.avg,"Best/NSS median": nss_loss.get_median(), "Best/NSS std": nss_loss.get_std(),
        "Best/SIM mean": sim_loss.avg,"Best/SIM median": sim_loss.get_median(), "Best/SIM std": sim_loss.get_std()})
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), args.model_val_path)
                    else:
                        torch.save(model.state_dict(), args.model_val_path)
                print()

    if args.lr_sched:
        scheduler.step()
