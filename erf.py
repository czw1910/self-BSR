import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from model.DRWBSN import DRWBSN
import torch
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
import cv2
import os
#这里随机拿100张测试集图像，放到一个文件夹中，img_dir是文件夹路径
img_dir = r"D:\Self-BSR\数据集\CVC09_900_100—第一版\val\noise_r5s5"
# img_dir = "./001"
images=os.listdir(img_dir)
import os
import torch
from torch.utils.data import DataLoader
from dataset.dataset import dataset_loader
from utils.utils import *
from utils.metrics import *
from options import get_train_options,get_train_dir,print_options
from model.DRWBSN import DRWBSN



def calculate_mean_and_std(img_dir):
    all_pixels = []
    
    for img_name in os.listdir(img_dir):
        if img_name.endswith('.png') or img_name.endswith('.jpg'):
            img_path = os.path.join(img_dir, img_name)
            image = Image.open(img_path).convert('L')
            img_arr = np.array(image) / 255.0  # 将像素值归一化
            all_pixels.append(img_arr)

    all_pixels = np.concatenate(all_pixels).ravel()
    mean = np.mean(all_pixels)
    std = np.std(all_pixels)
    
    return mean, std

if __name__ == '__main__':
    opt, parser = get_train_options()
    # opt.name = 'CVC09_r5s5_DRWBSN-b6c64_loss_MSE_S-e2_MSE_I_S-1_TVx_Ip-3e3_b8c128e2000'
    get_train_dir(opt)   
    print_options(opt, parser)
    model_option = "best_psnr"
    # model_option = "best_ssim"
    # model_option = "250"    
    Refine = True
    model_path = os.path.join( opt.checkpoints_dir,"epoch_"+model_option+".pth")
    print(model_path)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    model = DRWBSN(opt,Refine).to(device)
    model.load_state_dict(torch.load(model_path))

    model = model.eval()
    #定义输入图像的长宽，这里需要保证每张图像都要相同
    input_H, input_W = 480, 640
    #生成一个和输入图像大小相同的0矩阵，用于更新梯度
    heatmap = np.zeros([input_H, input_W])
    #打印一下模型，选择其中的一个层
    # print(model)

    #这里选择骨干网络的最后一个模块
    layer = model.tail_I[-1]
    # print(layer)


    def farward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    #为了简单，这里直接一张一张图来算，遍历文件夹中所有图像
    # 
    # mean, std = calculate_mean_and_std(img_dir)  
    # print(mean, std)

    for img in images:
        print(img)
        read_img = os.path.join(img_dir,img)
        image = Image.open(read_img)

        #图像预处理，将图像缩放到固定分辨率，并进行标准化
        # print(image.shape)
        image = image.resize((input_W, input_H))
        
        image = np.float32(image) / 255
        print(image.shape)
        # input_tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.7), (0.7))])(image)
        input_tensor = transforms.Compose([transforms.ToTensor()])(image)
        

        #添加batch维度
        input_tensor = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            model = model.cuda()
            input_tensor = input_tensor.cuda()
            
        #输入张量需要计算梯度
        input_tensor.requires_grad = True
        fmap_block = list()
        input_block = list()

        #对指定层获取特征图
        layer.register_forward_hook(farward_hook)

        #进行一次正向传播
        output = model(input_tensor)

        #特征图的channel维度算均值且去掉batch维度，得到二维张量
        feature_map = fmap_block[0].mean(dim=1,keepdim=False).squeeze()

        #对二维张量中心点（标量）进行backward
        feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)

        #对输入层的梯度求绝对值
        grad = torch.abs(input_tensor.grad)


        #梯度的channel维度算均值且去掉batch维度，得到二维张量，张量大小为输入图像大小
        grad = grad.mean(dim=1,keepdim=False).squeeze()

        # 确保与 heatmap 形状一致（640, 480）
        grad = grad.cpu().numpy()  # 使用转置使其形状变为 (640, 480)

        #累加所有图像的梯度，由于后面要进行归一化，这里可以不算均值
        heatmap = heatmap + grad


        cam = heatmap
        # torch.cuda.empty_cache()

    #对累加的梯度进行归一化

    cam = cam / cam.max()


    #可视化，蓝色值小，红色值大
    cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_INFERNO)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    Image.fromarray(cam)

    # 保存图像到指定路径
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    output_path = os.path.join(output_dir, 'T-1-2.png')
    cv2.imwrite(output_path, cam)

    # 或者使用 PIL 保存
    Image.fromarray(cam).save(output_path)

    # print(f"感受野图像已保存至: {output_path}")

