import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pywt

class DBSN_res(nn.Module):
    def __init__(self, opt, refine=False):
        super().__init__()
        self.opt = opt
        self.refine = refine
        self.experimrnt_dir = self.opt.experimrnt_dir        
        in_ch=self.opt.input_ch
        out_ch=self.opt.input_ch
        base_ch=self.opt.base_ch
        num_module=self.opt.bnum
        self.head = nn.Sequential(
                                nn.Conv2d(in_ch, base_ch, kernel_size=1),
                                nn.ReLU(inplace=True),)
        self.maskconv_I = nn.Sequential(
                        CentralMaskedConv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),            
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),                                
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),
                        nn.ReLU(inplace=True),
                        )
        self.branch_I = nn.Sequential( *[ResBlock(base_ch) for _ in range(num_module)])
        self.branch_S = nn.Sequential( *[ResBlock(base_ch) for _ in range(num_module)])  
        self.S2I = nn.Sequential( *[DCl(2, base_ch) for _ in range(2)])              
        self.branchI_tail = nn.Sequential(
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),
                        nn.ReLU(inplace=True),
                        )
        self.branchS_tail = nn.Sequential(
                        nn.Conv2d(base_ch, base_ch, kernel_size=1),
                        nn.ReLU(inplace=True),
                        )                   
        self.tail_I = nn.Sequential(
                                nn.Conv2d(base_ch,  base_ch,    kernel_size=1),
                                nn.ReLU(inplace=True),                                
                                nn.Conv2d(base_ch,  base_ch//2,    kernel_size=1),
                                nn.ReLU(inplace=True),                                
                                nn.Conv2d(base_ch//2, base_ch//2, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(base_ch//2,  out_ch, kernel_size=1),                                 
                                )
        self.tail_S = nn.Sequential(
                        nn.Conv2d(base_ch,  base_ch,    kernel_size=(3,1), stride=1, padding=(1,0)),
                        nn.ReLU(inplace=True),                                
                        nn.Conv2d(base_ch,  base_ch//2,    kernel_size=(3,1), stride=1, padding=(1,0)),
                        nn.ReLU(inplace=True),                                
                        nn.Conv2d(base_ch//2, base_ch//2, kernel_size=(3,1), stride=1, padding=(1,0)),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch//2,  out_ch, kernel_size=(3,1), stride=1, padding=(1,0)),                                 
                        )
        self.RWT = waveletDecomp(stride=1, c_channels=base_ch)
        self.draw_feature = False
    def forward(self, x):
        x_h = self.head(x)      
        if self.refine:
            x_h = F.interpolate(x_h,scale_factor=(2,1.6),mode='bilinear')                      
        x_mask = self.maskconv_I(x_h)
        if self.refine:
            x_mask = F.interpolate(x_mask,scale_factor=(1/2,1/1.6),mode='bilinear')  
        I_pd_b = self.branch_I(x_mask)
        S_pd_b = self.branch_S(x_mask)    
        S2I = self.S2I(S_pd_b)
        I_pd_b_en = I_pd_b + S2I

        I_b = self.branchI_tail(I_pd_b_en) 
        S_b = self.branchS_tail(S_pd_b)  

        stripe = self.tail_S(S_b)               
        img_clean = self.tail_I(I_b)
        stripe_ = torch.sum(stripe,dim=-2,keepdim=True)
        if self.draw_feature:
            draw_features(8,8,x_h.cpu().numpy(),self.experimrnt_dir,'x_h.png')
            draw_features(8,8,x_mask.cpu().numpy(),self.experimrnt_dir,'x_mask.png')
            draw_features(8,8,S2I.cpu().numpy(),self.experimrnt_dir,'S2I.png')             
            draw_features(8,8,I_pd_b_en.cpu().numpy(),self.experimrnt_dir,'I_pd_b_en.png')             
            draw_features(8,8,I_pd_b.cpu().numpy(),self.experimrnt_dir,'I_pd_b.png')
            draw_features(8,8,S_pd_b.cpu().numpy(),self.experimrnt_dir,'S_pd_b.png')
            draw_features(8,8,I_b.cpu().numpy(),self.experimrnt_dir,'I_b.png')
            draw_features(8,8,S_b.cpu().numpy(),self.experimrnt_dir,'S_b.png') 
            draw_features(1,1,stripe.cpu().numpy(),self.experimrnt_dir,'stripe.png')  
            draw_features(1,1,img_clean.cpu().numpy(),self.experimrnt_dir,'img_clean.png')  
        return img_clean,stripe,stripe_

class ResBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        ly = []
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride) ]
        ly += [ nn.ReLU(inplace=True) ]
        ly += [ nn.Conv2d(in_ch, in_ch, kernel_size=1) ]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH//2, kH//2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

class waveletDecomp(nn.Module):
    def __init__(self, stride=2, c_channels=1):
        super(waveletDecomp, self).__init__()

        self.stride = stride
        self.c_channels = c_channels

        wavelet = pywt.Wavelet('haar')
        dec_hi = torch.tensor(wavelet.dec_hi[::-1])
        dec_lo = torch.tensor(wavelet.dec_lo[::-1])

        self.filters_dec = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                                        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_dec = self.filters_dec.unsqueeze(1)
        self.filters_dec = self.filters_dec.repeat((self.c_channels,1,1,1))
        self.psize = int(self.filters_dec.size(3) / 2)

        rec_hi = torch.tensor(wavelet.rec_hi[::-1])
        rec_lo = torch.tensor(wavelet.rec_lo[::-1])
        self.filters_rec = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                                        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0).cuda()
        self.filters_rec = self.filters_rec.unsqueeze(0)
        self.filters_rec = self.filters_rec.repeat((self.c_channels,1,1,1))  
        self.filters_rec_transposed = torch.flip(self.filters_rec.permute(1, 0, 2, 3), [2, 3])

    def forward(self, x):
        if self.stride == 1:
            x = F.pad(x, (self.psize - 1, self.psize, self.psize - 1, self.psize), mode='replicate')
        coeff = F.conv2d(x, self.filters_dec, stride=self.stride, bias=None, padding=0, groups=self.c_channels)
        
        out = coeff / 2
        HL = out[:, 2::4, :, :].contiguous()
        return out,HL

    def inverse(self, x):
        # x = out
        if self.stride == 1:
            x = F.pad(x, (self.psize, self.psize - 1, self.psize, self.psize - 1), mode='replicate')

        if self.stride == 1:       
            coeff = F.conv2d(x, self.filters_rec, stride=self.stride, bias=None, padding=0, groups=self.c_channels)
        else:
            coeff = F.conv_transpose2d(x, self.filters_rec_transposed, stride=self.stride,
                                                       bias=None, padding=0)
        out = coeff

        return (out * self.stride ** 2) / 2


def pixel_shuffle_down_sampling_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    b, c, w, h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), 'reflect')
    unshuffled = unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 2, 3, 1, 4, 5).contiguous()
    unshuffled = unshuffled.view(-1, c, w // f + 2 * pad, h // f + 2 * pad).contiguous()
    return unshuffled
        
def pixel_shuffle_up_sampling_pd(x: torch.Tensor, f: int, pad: int = 0):

    b, c, w, h = x.shape
    b = b // (f * f)
    before_shuffle = x.view(b, f, f, c, w, h)
    before_shuffle = before_shuffle.permute(0, 3, 1, 2, 4, 5).contiguous()
    before_shuffle = before_shuffle.view(b, c*f*f, w, h)
    if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)


def draw_features(width,height,x,exp_path,name):#feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    save_path_imgs = os.path.join(exp_path,'feature')
    if not os.path.exists(save_path_imgs):
        os.makedirs(save_path_imgs)  # feature_map[2].shape     out of bounds
    print(x.shape)
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width*height):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1]之间，转换成0-255
        img=img.astype(np.uint8)  #转成unit8
        img=cv2.applyColorMap(img, cv2.COLORMAP_JET) #生成heat map
        img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i,width*height))
    fig.savefig(os.path.join(save_path_imgs,name), dpi=100)
    fig.clf()
    plt.close()