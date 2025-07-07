""" 关于图像评价指标的函数 """
import os
import cv2
import numpy as np
# import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# import lpips
# lpips_model = lpips.LPIPS(net="alex", verbose=False).cuda()

def compute_psnr_ssim(img_out, img_clean, data_range):
    if isinstance(img_out, torch.Tensor):
        img_out = img_out.squeeze().cpu().detach().numpy().astype(np.float32)
        img_clean = img_clean.squeeze().cpu().detach().numpy().astype(np.float32)
    # psnr = psnr_numpy(img_noise, img_clean)
    # ssim = calculate_ssim(img_noise, img_clean)
    psnr = peak_signal_noise_ratio(img_clean, img_out, data_range=data_range)
    ssim = structural_similarity(img_clean, img_out, data_range=data_range)
    return psnr, ssim

# def compute_psnr_ssim_lpips(img_out, img_clean): 
#     lpips_result = calculate_lpips(img_out,img_clean)   
#     psnr, ssim = compute_psnr_ssim(img_out,img_clean,1.)
#     return psnr, ssim, lpips_result

# def psnr_torch(tar_img, prd_img):
#     imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
#     rmse = (imdff ** 2).mean().sqrt()
#     ps = 20 * torch.log10(1 / rmse)
#     return ps

def psnr_numpy(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff ** 2.))
    if rmse == 0:
        rmse = 1
    ps = 20 * np.log10(255.0 / rmse)
    return ps


# def calculate_lpips(img1, img2):
#     if not isinstance(img1, torch.Tensor):
#         img1 = torch.from_numpy(img1).permute(2, 0, 1)
#         img2 = torch.from_numpy(img2).permute(2, 0, 1)
#     img1 = (img1 - img1.min()) / (img1.max() - img1.min())
#     img2 = (img2 - img2.min()) / (img2.max() - img2.min())
#     lpips_result = lpips_model.forward(img1, img2, normalize=True).mean().item()
#     return lpips_result


def ssim_numpy(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim_numpy(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_numpy(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_numpy(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# def calculate_dir2(dir1, dir2):
#     sum_psnr = 0
#     sum_ssim = 0
#     sum_lpips = 0
#     results = {"img": [], 'psnr': [], 'ssim': [], 'lpips': []}

#     name_list = os.listdir(dir1)
#     name_list.sort()
#     name_list2 = os.listdir(dir2)
#     name_list2.sort()
#     img_num = 0
#     for i, name in enumerate(name_list):
#         if name == "compare.png" or name == "calculate_result.png" or not name.endswith(".png"):
#             continue
#         img_num += 1
#         path1 = os.path.join(dir1, name)
#         path2 = os.path.join(dir2, name_list2[i])
#         img1 = cv2.imread(path1)
#         img2 = cv2.imread(path2)
#         psnr = psnr_numpy(img1, img2)
#         ssim = calculate_ssim(img1, img2)
#         # img1 = img1.transpose(2, 1, 0)
#         # img2 = img2.transpose(2, 1, 0)
#         # psnr, ssim = compute_psnr_ssim(img1, img2, 1.)
#         # lpips = calculate_lpips(img1, img2)
#         lpips = 0
#         results['img'].append(name)
#         results['psnr'].append(psnr)
#         results['ssim'].append(ssim * 100)
#         results['lpips'].append(lpips * 100)

#         sum_psnr += psnr
#         sum_ssim += ssim
#         sum_lpips += lpips
#         # print("name：{:.4f}\npsnr:{:.4f} ssim:{:.4f} lpips:{:.4f}\n-----".format(name, psnr, ssim, lpips))

#     ave_psnr = sum_psnr / img_num
#     ave_ssim = sum_ssim / img_num
#     ave_lpips = sum_lpips / img_num
#     print("*****")
#     print("Average psnr:{:.4f} ssim:{:.4f} lpips:{:.4f}".format(ave_psnr, ave_ssim, ave_lpips))
#     print("*****")

#     data_frame = pd.DataFrame(
#         data={'IMG_NAME': results['img'], 'PSNR': results['psnr'], 'SSIM': results['ssim'], 'LPIPS': results['lpips']},
#         index=range(1, img_num + 1))
#     result_save_path = os.path.join(dir1, "test_results.csv")
#     line_figure = data_frame.plot.line()
#     line_figure = line_figure.get_figure()
#     line_figure.savefig(os.path.join(dir1, 'calculate_result.png'))
#     data_frame.to_csv(result_save_path, index_label='num')


def calculate_dir(dir1, dir2):
    sum_psnr = 0
    sum_ssim = 0
    sum_lpips = 0
    results = {"img": [], 'psnr': [], 'ssim': [], 'lpips': []}

    name_list = os.listdir(dir1)
    name_list.sort()
    img_num = 0
    for name in name_list:
        if name == "compare.png" or name == "calculate_result.png" or not name.endswith(".png"):
            continue
        img_num += 1
        path1 = os.path.join(dir1, name)
        path2 = os.path.join(dir2, name)
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        psnr = psnr_numpy(img1, img2)
        ssim = calculate_ssim(img1, img2)
        # img1 = img1.transpose(2, 1, 0)
        # img2 = img2.transpose(2, 1, 0)
        # psnr, ssim = compute_psnr_ssim(img1, img2, 1.)
        # lpips = calculate_lpips(img1, img2)
        lpips = 0
        results['img'].append(name)
        results['psnr'].append(psnr)
        results['ssim'].append(ssim * 100)
        results['lpips'].append(lpips * 100)

        sum_psnr += psnr
        sum_ssim += ssim
        sum_lpips += lpips
        # print("name：{:.4f}\npsnr:{:.4f} ssim:{:.4f} lpips:{:.4f}\n-----".format(name, psnr, ssim, lpips))

    ave_psnr = sum_psnr / img_num
    ave_ssim = sum_ssim / img_num
    ave_lpips = sum_lpips / img_num
    print("*****")
    print("Average psnr:{:.4f} ssim:{:.4f} lpips:{:.4f}".format(ave_psnr, ave_ssim, ave_lpips))
    print("*****")



if __name__ == "__main__":
    # dir1_list = ["/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa_img-L1_l1b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa_img-content_l1b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa_img-L1-content_l1b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa_img-L1-content_D-video_l1b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa-tem-r_cb1x1_img-1L1-1content_l5b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa-tem-r_cb1x1_img-1L1-3content_l5b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa-tem-r_wf3_img-1L1-3content_l5b8c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa-tem-bir_cb1x1_img-1l1-3content_l5b4c128e200",
    #              "/home/nvidia/05_data/02_NIR_dataset/Ablation_experiment/NIR_sim1_gen_spa-tem-bir_wf2_img-1L1-3content_l5b4c128e200"]
    # dir1_list = [os.path.join(dir1, "val", "noise_gen", "scene7") for dir1 in dir1_list]
    # dir1_list = ["/home/nvidia/05_data/02_NIR_dataset/generate/NIR_gen_sim1_RecycleGAN/val/noise_gen/scene7",
    #              "/home/nvidia/05_data/02_NIR_dataset/generate/NIR_gen_sim1_SCGAN/val/noise_gen/scene7",
    #              "/home/nvidia/05_data/02_NIR_dataset/generate/NIR_gen_sim1_CycleGAN/val/noise_gen/scene7"]

    # dir1_list = ["/home/nvidia/01_deeplearning/01_denoise/AllNet/checkpoints/fastdvdnet_NIR_gen_sim1_RecycleGAN/test/ckpt_epoch400_sim_noise_scene7",
    #              "/home/nvidia/01_deeplearning/01_denoise/AllNet/checkpoints/fastdvdnet_NIR_gen_sim1_SCGAN/test/ckpt_epoch400_sim_noise_scene7",
    #              "/home/nvidia/01_deeplearning/01_denoise/AllNet/checkpoints/fastdvdnet_NIR_gen_sim1_CycleGAN/test/ckpt_epoch400_sim_noise_scene7",
    #              "/home/nvidia/05_data/02_NIR_dataset/generate/01_sim_compare/denoise_scene7/UDVD",
    #              "/home/nvidia/05_data/02_NIR_dataset/generate/01_sim_compare/denoise_scene7/AP_BSN"]
    dir1_list = ["/home/nvidia/01_deeplearning/01_denoise/AllNet/checkpoints/fastdvdnet_NIR_gen_sim1_CycleGAN/test/ckpt_epoch400_sim_noise_scene7"]
    dir2 = "/home/nvidia/05_data/02_NIR_dataset/NIR_sim1/val/clean/scene7"
    # dir2 = "/home/nvidia/05_data/02_NIR_dataset/NIR_sim1/train/clean/scene1"
    for dir1 in dir1_list:
        print("dir1: {}".format(dir1))
        print("dir2: {}".format(dir2))
        calculate_dir(dir1, dir2)
        # cal_KL_dir(dir1, dir2)
    print("Finished!")
