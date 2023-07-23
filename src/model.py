import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from  scipy import ndimage

sys.path.append('./PNAS/')
from PNASnet import *
from genotypes import PNASNet
import torch.nn.functional as nnf
import numpy as np


   
class PNASModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(PNASModel, self).__init__()
        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
        if load_weight:
            self.pnas.load_state_dict(torch.load(self.path))

        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)

        x = self.deconv_layer4(x)
        
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
     #   print("PNAS pred actual pnas:", x.mean(),x.min(), x.max(), x.sum())

        return x

class PNASVolModellast(nn.Module):

    def __init__(self, time_slices, num_channels=3, train_enc=False, load_weight=1):
            super(PNASVolModellast, self).__init__()

            self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
            if load_weight:            
                state_dict = torch.load(self.path)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'module'  in k:
                        k = 'module.pnas.' + k
                    else:
                        k = k.replace('pnas.', '')
                    new_state_dict[k] = v
                self.pnas.load_state_dict(new_state_dict, strict=False)
               

            for param in self.pnas.parameters():
                param.requires_grad = train_enc

            self.padding = nn.ConstantPad2d((0,1,0,1),0)
            self.drop_path_prob = 0

            self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

            self.deconv_layer0 = nn.Sequential(
                nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )

            self.deconv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )

            self.deconv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 32, out_channels = time_slices, kernel_size = 3, padding = 1, bias = True),
                nn.Sigmoid()
            )
        
    def forward(self, images):
        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)

        x = self.deconv_layer4(x)

        x = self.deconv_layer5(x)
        x = x / x.max()

        return x , [out1,out2,out3,out4,out5]   
    

class PNASBoostedModelMultiLevel(nn.Module):

    def __init__(self, device, model_path, model_vol_path, time_slices, train_model=False, selected_slices=""):
        super(PNASBoostedModelMultiLevel, self).__init__()

        
        self.selected_slices = selected_slices

            
        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels = 512+2160+6, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        self.deconv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels = 1080+256+6, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        self.deconv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels = 540+6, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        self.deconv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels = 192+6, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        

        self.deconv_mix = nn.Sequential(
            nn.Conv2d(in_channels = 128+6 , out_channels = 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        model_vol = PNASVolModellast(time_slices=5, load_weight=0) #change this to time slices
        model_vol = nn.DataParallel(model_vol).cuda()
        state_dict = torch.load(model_path)
        vol_state_dict = OrderedDict()
        sal_state_dict = OrderedDict()
        smm_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if 'pnas_vol'  in k:

                k = k.replace('pnas_vol.module.', '')
                vol_state_dict[k] = v
            elif 'pnas_sal'  in k:
                k = k.replace('pnas_sal.module.', '')
                sal_state_dict[k] = v
            else:
                smm_state_dict[k] = v
                
        self.load_state_dict(smm_state_dict)
        model_vol.load_state_dict(vol_state_dict)
        self.pnas_vol = nn.DataParallel(model_vol).cuda()

        for param in self.pnas_vol.parameters():
            param.requires_grad = False


        model = PNASModel(load_weight=0)
        model = nn.DataParallel(model).cuda()

        model.load_state_dict(sal_state_dict, strict=True)
        self.pnas_sal = nn.DataParallel(model).to(device)

        for param in self.pnas_sal.parameters():
            param.requires_grad = False #train_model

    def forward(self, images):
      #  print("IMAGES", images.shape)

        pnas_pred = self.pnas_sal(images).unsqueeze(1) 
        pnas_vol_pred , outs = self.pnas_vol(images)

        out1 , out2, out3, out4, out5 = outs
        #print(pnas_vol_pred.shape)
        x_maps = torch.cat((pnas_pred, pnas_vol_pred), 1)

        x = torch.cat((out5,out4), 1)
        x_maps16 = nnf.interpolate(x_maps, size=(16, 16), mode='bicubic', align_corners=False)

        x = torch.cat((x,x_maps16), 1)

        x = self.deconv_layer1(x)
        x = torch.cat((x,out3), 1)
        x_maps32 = nnf.interpolate(x_maps, size=(32, 32), mode='bicubic', align_corners=False)
        x = torch.cat((x,x_maps32), 1)

        x = self.deconv_layer2(x)
        x = torch.cat((x,out2), 1)
        x_maps64 = nnf.interpolate(x_maps, size=(64, 64), mode='bicubic', align_corners=False)
        x = torch.cat((x,x_maps64), 1)

        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)
        x_maps128 = nnf.interpolate(x_maps, size=(128, 128), mode='bicubic', align_corners=False)

        x = torch.cat((x,x_maps128), 1)

        x = self.deconv_layer4(x)
        x = torch.cat((x,x_maps), 1)
        
        x = self.deconv_mix(x)
      
        x = x.squeeze(1)

        return x, pnas_vol_pred
    
