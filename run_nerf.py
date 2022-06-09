import os, sys
import numpy as np
import imageio
import json
import random
import time
import datetime
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import lpips as lp
from run_nerf_helpers import *
from load_real_llff import load_real_llff_data
from load_syn_llff import load_syn_llff_data
from config import config_parser
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        outputs = []
        ln_x = []
        for i in range(0, inputs.shape[0], chunk):
            out, inp = fn(inputs[i:i+chunk])
            outputs.append(out)
            ln_x.append(inp)
        return torch.cat(outputs, 0), ln_x
    return ret


def run_network(inputs, viewdirs, exptimes, fn, embed_fn, embeddirs_fn, embedexps_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)

        input_exps = torch.broadcast_to(exptimes[:,None,:], [exptimes.shape[0], inputs.shape[1], 1])
        input_exps_flat = torch.reshape(input_exps, [-1, input_exps.shape[-1]])
        # embedded_exps = embedexps_fn(input_exps_flat)

        embedded = torch.cat([embedded, embedded_dirs, input_exps_flat], -1) # without exposure embedding

    outputs_flat, lnx = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs, lnx


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    lnx_list = []
    for i in range(0, rays_flat.shape[0], chunk):
        ret, lnx = render_rays(rays_flat[i:i+chunk], **kwargs)
        lnx_list.append(lnx)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret, lnx_list


def render(H, W, K, chunk=1024*32, rays=None, exps=None, c2w=None, ndc=True,
           near=0., far=1., use_viewdirs=False, c2w_staticcam=None, **kwargs):
    """Render rays
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
        exps = torch.broadcast_to(exps[None, None, :], [H, W, 1])

    else:
        # use provided ray batch
        rays_o, rays_d = rays[:,:3], rays[:,3:]

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    exps = torch.reshape(exps, [-1, 1]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs, exps], -1)

    # Render and reshape
    all_ret, lnx = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_h_map', 'rgb_l_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict] + [lnx]


def render_path(render_poses, render_exps, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, i_test=[], datatype='real_llff'):
    
    H, W, focal = hwf
    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    t = time.time()
    print('START RENDERING...')

    rgbs = []; disps = []; rgbs_h = []
    np_oe_psnr = []; np_ne_psnr = []; op_oe_psnr = []; op_ne_psnr = []
    np_oe_ssim = []; np_ne_ssim = []; op_oe_ssim = []; op_ne_ssim = []
    np_oe_lpips = []; np_ne_lpips = []; op_oe_lpips = []; op_ne_lpips = []

    for i, c2w in enumerate(render_poses):
        idx = i
        if len(i_test) != 0:
            idx = i_test[i]

        t = time.time()
        rgb_h, rgb, disp, acc, _, _ = render(H, W, K, chunk=chunk, exps=render_exps[i,:], c2w=c2w[:3,:4], **render_kwargs)
        rgbs_h.append(rgb_h.cpu().numpy())
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        # save the evaluation results and render views
        if gt_imgs is not None and render_factor==0:
            fake_image = rgb.cpu().numpy()
            real_image = gt_imgs[i].cpu().numpy()
            psnr = compare_psnr(fake_image, real_image, data_range=1)
            ssim = compare_ssim(fake_image, real_image, data_range=1.0, multichannel=True)

            loss_fn_alex = lp.LPIPS(net='alex')
            lp_tensor1 = (rgb*2. - 1.).permute(2,0,1)[None, ...]
            lp_tensor2 = (gt_imgs[i]*2. - 1.).permute(2,0,1)[None, ...]
            lpips = loss_fn_alex(lp_tensor1, lp_tensor2)
            lpips = np.squeeze(lpips.cpu().numpy())

            filename = os.path.join(savedir, '{:03d}_gt.png'.format(idx))
            imageio.imwrite(filename, to8b(real_image))

            print('Frame {}: PSNR={:.3f} SSIM={:.3f} LPIPS={:.3f}'.format(i, psnr, ssim, lpips*100))
            if datatype == 'real_llff':
                if idx % 10 >= 5:
                    np_oe_psnr.append(psnr)
                    np_oe_ssim.append(ssim)
                    np_oe_lpips.append(lpips)
                    with open(os.path.join(savedir, 'eval_real.csv'),"a") as csvfile: 
                        writer = csv.writer(csvfile)
                        writer.writerow([idx, psnr, ssim, lpips])
            elif datatype == 'syn_llff':
                np_oe_psnr.append(psnr)
                np_oe_ssim.append(ssim)
                np_oe_lpips.append(lpips)
                with open(os.path.join(savedir, 'eval_syn.csv'),"a") as csvfile: 
                    writer = csv.writer(csvfile)
                    writer.writerow([idx, psnr, ssim, lpips])
    
        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb8_h_tm = tonemap(rgbs_h[-1] / np.max(rgbs_h[-1]))

            filename1 = os.path.join(savedir, '{:03d}_ours.png'.format(idx))
            filename2 = os.path.join(savedir, '{:03d}_ours_h_tm.png'.format(idx))
            filename3 = os.path.join(savedir, '{:03d}_ours_hdr.exr'.format(idx))

            imageio.imwrite(filename1, rgb8)
            imageio.imwrite(filename2, rgb8_h_tm)
            imageio.imwrite(filename3, rgbs_h[-1])

        print('View: {:03d}, Render Time: {:.03f}'.format(i, time.time() - t))
    
    if gt_imgs is not None and render_factor==0:
        if datatype == 'real_llff':
            all_psnr = np_oe_psnr
            all_ssim = np_oe_ssim 
            all_lpips = np_oe_lpips
            with open(os.path.join(savedir, 'eval_real.csv'),"a") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerows([['np_0', np.mean(np_oe_psnr[0::5]), np.mean(np_oe_ssim[0::5]), np.mean(np_oe_lpips[0::5])],
                                  ['np_1', np.mean(np_oe_psnr[1::5]), np.mean(np_oe_ssim[1::5]), np.mean(np_oe_lpips[1::5])],
                                  ['np_2', np.mean(np_oe_psnr[2::5]), np.mean(np_oe_ssim[2::5]), np.mean(np_oe_lpips[2::5])],
                                  ['np_3', np.mean(np_oe_psnr[3::5]), np.mean(np_oe_ssim[3::5]), np.mean(np_oe_lpips[3::5])],
                                  ['np_4', np.mean(np_oe_psnr[4::5]), np.mean(np_oe_ssim[4::5]), np.mean(np_oe_lpips[4::5])],
                                  ['all mean', np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips)]])
        elif datatype == 'syn_llff':
            all_psnr = np_oe_psnr
            all_ssim = np_oe_ssim 
            all_lpips = np_oe_lpips
            with open(os.path.join(savedir, 'eval_syn.csv'),"a") as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerows([['np_0', np.mean(np_oe_psnr[0::5]), np.mean(np_oe_ssim[0::5]), np.mean(np_oe_lpips[0::5])],
                                  ['np_1', np.mean(np_oe_psnr[1::5]), np.mean(np_oe_ssim[1::5]), np.mean(np_oe_lpips[1::5])],
                                  ['np_2', np.mean(np_oe_psnr[2::5]), np.mean(np_oe_ssim[2::5]), np.mean(np_oe_lpips[2::5])],
                                  ['np_3', np.mean(np_oe_psnr[3::5]), np.mean(np_oe_ssim[3::5]), np.mean(np_oe_lpips[3::5])],
                                  ['np_4', np.mean(np_oe_psnr[4::5]), np.mean(np_oe_ssim[4::5]), np.mean(np_oe_lpips[4::5])],
                                  ['all mean', np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips)]])

    rgbs = np.stack(rgbs, 0)
    rgbs_h = np.stack(rgbs_h, 0)
    disps = np.stack(disps, 0)

    return rgbs_h, rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed)

    # device_ids = [0, 1]
    input_ch_views = 0
    input_ch_exps = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)
        embedexps_fn, input_ch_exps = get_embedder(args.multires_views, 1, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, exptimes, network_fn : run_network(inputs, viewdirs, exptimes, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedexps_fn=embedexps_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))


    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if (args.dataset_type != 'real_llff' and args.dataset_type != 'syn_llff') or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb_h = torch.exp(raw[..., :3])
    rgb_l = torch.sigmoid(raw[..., 3:6])
    
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,-1].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,-1].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,-1] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1] + 1e-10
    rgb_h_map = torch.sum(weights[...,None] * rgb_h, -2)  # [N_rays, 3]
    rgb_l_map = torch.sum(weights[...,None] * rgb_l, -2) 

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_h_map = rgb_h_map + (1.-acc_map[...,None])
        rgb_l_map = rgb_l_map + (1.-acc_map[...,None])

    return rgb_h_map, rgb_l_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False, perturb=0., 
                N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False, pytest=False):
    """Volumetric rendering.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-4:-1] if ray_batch.shape[-1] > 8 else None
    exptims = ray_batch[:,-1:]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # raw = run_network(pts)
    raw, _ = network_query_fn(pts, viewdirs, exptims, network_fn)
    rgb_h_map, rgb_l_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_h_map_0, rgb_l_map_0, disp_map_0, acc_map_0 = rgb_h_map, rgb_l_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        # raw = run_network(pts, fn=run_fn)
        raw, lnx = network_query_fn(pts, viewdirs, exptims, run_fn)

        rgb_h_map, rgb_l_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_h_map' : rgb_h_map, 'rgb_l_map' : rgb_l_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        # ret['lnx'] = lnx
        ret['rgb_h0'] = rgb_h_map_0
        ret['rgb_l0'] = rgb_l_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret, lnx


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load forward-facing data
    K = None
    if args.dataset_type == 'real_llff':
        images, poses, bds, exps_source, render_poses, render_exps, i_test = load_real_llff_data(args.datadir, args.factor,
                                                                                                 recenter=True, bd_factor=.75, spherify=args.spherify,
                                                                                                 max_exp=args.max_exp, min_exp=args.min_exp)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        exps = exps_source
        print('Loaded real llff:', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]
        
        # randomly select an exposure from {t_1, t_3, t_5} for each input view
        elif args.llffhold == 0: 
            print('Random select images for training.')
            np.random.seed(args.random_seed)
            i_train = []
            exp_num = 5
            for i in range(images.shape[0] // (exp_num*2) + 1):
                step = i*exp_num*2
                i_train.append(np.random.choice([0+step, 2+step, 4+step], 1, replace=False))
            i_train = np.sort(np.array(i_train).reshape([-1]))
            i_test = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_train)])

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])
        i_train = i_train[::args.train_sample]

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'syn_llff':
        images, poses, exps_source, render_poses, render_exps, hwf, i_split = load_syn_llff_data(args.datadir, args.half_res, args.testskip, 
                                                                                max_exp=args.max_exp, min_exp=args.min_exp, near_depth=args.near_depth, rand_seed=args.random_seed, render_size=args.render_size)
        print('Loaded synthetic llff:', images.shape, render_poses.shape, hwf, args.datadir)
        exps = exps_source
        i_train, i_test = i_split
        i_val = i_test

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]
        
        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = args.near_depth
            far = args.near_depth + 8
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_exps = np.array(exps_source[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_exps = torch.Tensor(render_exps).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            idx_test = np.array([], dtype=np.int64)
            if args.render_test:
                # render_test switches to test poses
                images = torch.Tensor(images[i_test]).to(device)
                idx_test = i_test
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, args.render_out_path + '_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('Test poses shape:', render_poses.shape)

            rgbs_h, rgbs, disps = render_path(render_poses, render_exps, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, 
                                              render_factor=args.render_factor, i_test=idx_test, datatype=args.dataset_type)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video_' + args.expname + '_' + args.render_out_path +'.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'video_tm_'+ args.expname + '_' + args.render_out_path +'.mp4'), tonemap(rgbs_h/np.mean(np.max(rgbs_h, 0))), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'disp_'+ args.expname + '_' + args.render_out_path +'.mp4'), to8b(disps / np.max(rgbs_h)), fps=30, quality=8)
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    exps =  np.tile(exps[:,None,None,:], [1, H, W, 1])
    if use_batching:
        # For random ray batching
        print('Get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [rays_rgb.shape[0], H, W, -1])
        # exps = np.broadcast_to(exps[:,None,None,:], [rays_rgb.shape[0], H, W, 1])
        rays_rgb_exps = np.concatenate([rays_rgb, exps], -1)
        rays_rgb_exps = np.stack([rays_rgb_exps[i] for i in i_train], 0) # train images only
        rays_rgb_exps = np.reshape(rays_rgb_exps, [-1, 10]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_exps = rays_rgb_exps.astype(np.float32)
        print('Shuffle rays')
        np.random.shuffle(rays_rgb_exps)
        i_batch = 0

    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    exps = torch.Tensor(exps).to(device)

    if use_batching:
        rays_rgb_exps = torch.Tensor(rays_rgb_exps).to(device)

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb_exps[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch_rays, batch_exps, target_s = batch[:, :6], batch[:, 9:], batch[:, 6:9]
            i_batch += N_rand
            if i_batch >= rays_rgb_exps.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb_exps.shape[0])
                rays_rgb_exps = rays_rgb_exps[rand_idx]
                i_batch = 0
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]
            exp = exps[img_i,...]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.cat([rays_o, rays_d], -1)
                batch_exps = exp[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb_h, rgb, disp, acc, extras, lnx = render(H, W, K, chunk=args.chunk, rays=batch_rays, exps=batch_exps,
                                                    verbose=i < 10, retraw=True, **render_kwargs_train)

        optimizer.zero_grad()

        ### f(0) = z_mid
        unit_exp_loss = point_constraint(render_kwargs_train['network_fn'], args.fixed_value)
        img_loss = img2mse(rgb, target_s, 1)
        loss_backward = img_loss + unit_exp_loss*0.5

        trans = extras['raw'][...,-1]
        loss = img_loss # loss for output
        psnr = mse2psnr(img_loss)

        if 'rgb_l0' in extras:
            loss_backward0 = img2mse(extras['rgb_l0'], target_s, 1)
            img_loss0 = img2mse(extras['rgb_l0'], target_s, 1)
            unit_exp_loss_f = point_constraint(render_kwargs_train['network_fine'], args.fixed_value)
            loss_backward = loss_backward + loss_backward0 + unit_exp_loss_f*0.5
            loss = loss + img_loss0
            psnr0 = mse2psnr(img2mse(extras['rgb_l0'], target_s, 1))

        loss_backward.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        print('Step: {:d}/{:d}, Loss: {:.6f}, Time: {:.3f}, ReTime: '.format(global_step, N_iters-1, loss, dt) + 
                    str(datetime.timedelta(seconds=int(dt*(N_iters - global_step)))))
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict':  optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs_h, rgbs, disps = render_path(render_poses, render_exps, hwf, K, args.chunk, render_kwargs_test, datatype=args.dataset_type)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'rgb_tm.mp4',tonemap(rgbs_h/np.mean(np.max(rgbs_h, 0) + 1)), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            # draw_CRF(testsavedir, render_kwargs_test['network_fine'])
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), torch.Tensor(exps_source[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, 
                            gt_imgs=images[i_test], savedir=testsavedir, i_test=i_test, datatype=args.dataset_type)
            print('Saved test set')

        if i%args.i_print==0:
            # print(expname, i, psnr.item(), loss.item(), global_step)
            # print('iter time {:.05f}'.format(dt))

            writer.add_scalar('loss', loss.item(), global_step=global_step)
            writer.add_scalar('psnr', psnr.item(), global_step=global_step)
            writer.add_histogram('tran', trans, global_step=global_step)

            if args.N_importance > 0:
                writer.add_scalar('psnr0', psnr0.item(), global_step=global_step)

            if i%args.i_img==0:
                # Log a rendered validation view to Tensorboard
                img_i = i_val[4] # to reduce time consumption, we only select a single view for validation
                target = images[img_i]
                pose = poses[img_i, :3,:4]
                exp = exps_source[img_i,:]
                render_kwargs_test['network_fine'].eval()
                with torch.no_grad():
                    rgb_h, rgb, disp, acc, extras, _ = render(H, W, K, chunk=args.chunk, exps=torch.Tensor(exp).to(device), c2w=torch.Tensor(pose).to(device),
                                                        **render_kwargs_test)
                render_kwargs_test['network_fine'].train()

                psnr = mse2psnr(img2mse(rgb, target, 1))
                writer.add_scalar('psnr_holdout', psnr.item(), global_step=global_step)

                # writer.add_image('rgb', to8b(rgb.cpu().numpy()), global_step=global_step, dataformats='HWC')
                # writer.add_image('disp', disp, global_step=global_step, dataformats='HW')
                # writer.add_image('acc', acc, global_step=global_step, dataformats='HW')
                # writer.add_image('rgb_holdout_gt', target, global_step=global_step, dataformats='HWC')
                # if args.N_importance > 0:
                #     writer.add_image('rgb_l0', to8b(extras['rgb_l0'].cpu().numpy()), global_step=global_step, dataformats='HWC')
                #     writer.add_image('disp0', extras['disp0'], global_step=global_step, dataformats='HW')
                #     writer.add_image('z_std', extras['z_std'], global_step=global_step, dataformats='HW')
        
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor') # torch.cuda.FloatTensor | torch.cuda.DoubleTensor | torch.cuda.HalfTensor

    train()
