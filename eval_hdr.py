from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import imageio
import os
import numpy as np
import cv2
import csv
import torch 
import lpips as lp

tonemap = lambda x : (np.log(np.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1)).astype(np.float32)

if __name__ == '__main__':
    
    scene_list = ['sponza_new3', 'sofa_ours', 'dogroom_new', 'desk_ours', 'chair_ours2', 'bear_ours', 'bathroom_ours2'] # for ours method 
    gt_list = ['sponza', 'sofa', 'dogroom', 'desk', 'chair', 'bear', 'bathroom', 'diningroom']

    gt_root = '/apdcephfs/private_xanderhuang/our-dataset/syn-data/'
    scene_root = '/apdcephfs/private_xanderhuang/our-logs/'
    
    out_root = '/apdcephfs/private_xanderhuang/our-logs/'


    loss_fn_alex = lp.LPIPS(net='alex') # best forward scores
    loss_fn_vgg = lp.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    for i, scene in enumerate(scene_list):
        psnr_list = []
        ssim_list = []
        lp_list = []
        input_path = os.path.join(scene_root, scene, 'testset_200000')
        savedir = os.path.join(input_path, 'eval_hdr2')
        os.makedirs(savedir, exist_ok=True)
        file_list = os.listdir(input_path)
        file_list = sorted([f for f in file_list if f[-4:] == '.exr'])
        for j, file_name in enumerate(file_list[::5]):
            # ground truth
            gt_path = os.path.join(gt_root, gt_list[i], 'test_hdr/hdr_%03d.exr'%j)
            gt_hdr_source = imageio.imread(gt_path, 'exr')[..., 0:3]
            gt_hdr_source = cv2.resize(gt_hdr_source, (400, 400), interpolation=cv2.INTER_AREA)
            gt_hdr_normal = gt_hdr_source / np.max(gt_hdr_source)
            gt_hdr_tm = tonemap(gt_hdr_normal)
            imageio.imwrite(os.path.join(savedir, '%03d_gt.png'%j), (gt_hdr_tm * 255).astype(np.uint8))

            # our pre hdr images, use the max value of GT for normalization.
            pre_path = os.path.join(input_path, file_name)
            pre_hdr_normal = np.clip(imageio.imread(pre_path, 'exr')[..., 0:3] / np.max(gt_hdr_source), 0, 1) # 
        
            pre_hdr_tm = tonemap(pre_hdr_normal)
            imageio.imwrite(os.path.join(savedir, '%03d_pre.png'%j), (pre_hdr_tm * 255).astype(np.uint8))

            psnr = compare_psnr(gt_hdr_tm, pre_hdr_tm, data_range=1)
            ssim = compare_ssim(gt_hdr_tm, pre_hdr_tm, data_range=1.0, multichannel=True)
            psnr_list.append(psnr)
            ssim_list.append(ssim)

            gt_tensor = torch.Tensor(gt_hdr_tm * 2 - 1.).to('cpu').permute(2,0,1)[None, ...]
            pre_tensor = torch.Tensor(pre_hdr_tm * 2 - 1.).to('cpu').permute(2,0,1)[None, ...]
 
            lpips = loss_fn_alex(gt_tensor, pre_tensor)
            lpips = float(np.squeeze(lpips.detach().numpy()))
            lp_list.append(lpips)
            with open(os.path.join(savedir, 'eval_hdr.csv'),"a") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow([j, psnr, ssim, lpips])

            print([psnr, ssim, lpips])

            # Drago's tonemap
            tonemapDrago = cv2.createTonemapDrago(1.5, 2, 0.3)
            gt_d = tonemapDrago.process(gt_hdr_normal)
            pre_d = tonemapDrago.process(pre_hdr_normal)
            imageio.imwrite(os.path.join(savedir, '%03d_gt_drago.png'%j), (gt_d * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(savedir, '%03d_pre_drago.png'%j), (pre_d * 255).astype(np.uint8))

            # Durand's tonemap
            tonemapDurand = cv2.createTonemap(2.2)
            gt_d = tonemapDurand.process(gt_hdr_normal)
            pre_d = tonemapDurand.process(pre_hdr_normal)
            imageio.imwrite(os.path.join(savedir, '%03d_gt_gamma.png'%j), (gt_d * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(savedir, '%03d_pre_gamma.png'%j), (pre_d * 255).astype(np.uint8))

            # Reinhard's tonemap
            tonemapReinhard= cv2.createTonemapReinhard(2.2, 0.5, 0.5 ,0) # 2.2 1
            gt_d = tonemapReinhard.process(gt_hdr_normal)
            pre_d = tonemapReinhard.process(pre_hdr_normal)
            imageio.imwrite(os.path.join(savedir, '%03d_gt_reinhard.png'%j), (gt_d * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(savedir, '%03d_pre_reinhard.png'%j), (pre_d * 255).astype(np.uint8))

            # Mantiuk's tonemap
            tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 0.5)
            gt_d = tonemapMantiuk.process(gt_hdr_normal)
            pre_d = tonemapMantiuk.process(pre_hdr_normal)
            imageio.imwrite(os.path.join(savedir, '%03d_gt_mantiuk.png'%j), (gt_d * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(savedir, '%03d_pre_mantiuk.png'%j), (pre_d * 255).astype(np.uint8))


        with open(os.path.join(savedir, 'eval_hdr.csv'),"a") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow(['mean', np.mean(psnr_list), np.mean(ssim_list), np.mean(lp_list)])
        print('Done with ' + scene)
