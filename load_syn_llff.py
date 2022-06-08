import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def poses_avg(poses):
    bottom = np.reshape([0,0,0,1.], [1,4])
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), bottom], 0)
    
    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    bottom = np.reshape([0,0,0,1.], [1,4])
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), bottom], 0))
    return render_poses
    

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_

    return poses


def load_syn_llff_data(basedir, half_res=False, testskip=1, bd_factor=0.75, max_exp=1, min_exp=1, near_depth=4.0, rand_seed=1, render_size=30):
    np.random.seed(rand_seed)
    splits = ['train', 'test']
    metas = {}
    exps_metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
        with open(os.path.join(basedir, 'exposure_{}.json'.format(s)), 'r') as fp:
            exps_metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_exps = []
    counts = [0]
    num_exps = 5
    for s in splits:
        meta = metas[s]
        exps_meta = exps_metas[s]
        imgs = []
        poses = []
        exps = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            if s == 'train':
                idx = np.random.choice([0, 2, 4]) # randomly select an exposure from {t_1, t_3, t_5} for each input view
                fname = os.path.join(basedir, frame['file_path'] + '_%d.png' % idx)
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame['transform_matrix']))
                exps.append(np.float(exps_meta[frame['file_path'] + '_%d.png' % idx]))
            if s == 'test':
                for i in range(num_exps):
                    fname = os.path.join(basedir, frame['file_path'] + '_%d.png' % i)
                    imgs.append(imageio.imread(fname))
                    poses.append(np.array(frame['transform_matrix']))
                    exps.append(np.float(exps_meta[frame['file_path'] + '_%d.png' % i]))

        imgs = (np.array(imgs) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        exps = np.array(exps).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_exps.append(exps)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(2)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    sc = 1. if bd_factor is None else 1./(near_depth * bd_factor)
    poses[:, :3, 3] *= sc
    near_depth *= sc
    poses = recenter_poses(poses)
    exps = np.concatenate(all_exps, 0).reshape([-1, 1])
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3,:4])

    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    zdelta = near_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), render_size, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2

    # Generate poses and exposures for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    render_exps = np.linspace(min_exp, max_exp, N_views//2) # the exposure denotes exposure value (EV) 
    render_exps = 2 ** render_exps
    render_exps = np.concatenate([render_exps, render_exps[::-1]])
    render_exps = np.reshape(render_exps, [-1, 1]).astype(np.float32)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, exps, render_poses, render_exps, [H, W, focal], i_split