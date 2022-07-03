import torch
import os
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import scipy.io as scio

# Misc
img2mse = lambda x, y, z : torch.mean((x - y) ** 2 * z)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
tonemap = lambda x : (np.log(np.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1) * 255).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, in_dims, i=0):
    if i == -1:
        return nn.Identity(), in_dims
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : in_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_exps=1, output_ch=7, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_exps = input_ch_exps
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        self.exps_linears_r = nn.ModuleList([nn.Linear(1, W//2)])
        self.exps_linears_g = nn.ModuleList([nn.Linear(1, W//2)])
        self.exps_linears_b = nn.ModuleList([nn.Linear(1, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
            self.b_l_linner = nn.Linear(W//2, 1)
            self.r_l_linner = nn.Linear(W//2, 1)
            self.g_l_linner = nn.Linear(W//2, 1)
            self.rgb_l_linner = nn.Linear(W//2, 1)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views, input_exps = torch.split(x, [self.input_ch, self.input_ch_views, self.input_ch_exps], dim=-1)
        h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            spilit = True
            rgb_h = self.rgb_linear(h)
            if spilit:
                ## Split RGB channels
                r_h_s = rgb_h[:,0:1] + torch.log(input_exps)
                g_h_s = rgb_h[:,1:2] + torch.log(input_exps)
                b_h_s = rgb_h[:,2:3] + torch.log(input_exps)

                lnx = torch.cat([r_h_s, g_h_s, b_h_s], -1)

                r_h = lnx[:,0:1]
                g_h = lnx[:,1:2]
                b_h = lnx[:,2:3]
                
                for i, l in enumerate(self.exps_linears_r):
                    r_h = self.exps_linears_r[i](r_h)
                    r_h = F.relu(r_h)
                r_l = self.r_l_linner(r_h)

                for i, l in enumerate(self.exps_linears_g):
                    g_h = self.exps_linears_g[i](g_h)
                    g_h = F.relu(g_h)
                g_l = self.g_l_linner(g_h)

                for i, l in enumerate(self.exps_linears_b):
                    b_h = self.exps_linears_b[i](b_h)
                    b_h = F.relu(b_h)
                b_l = self.b_l_linner(b_h)

                rgb_l = torch.cat([r_l, g_l, b_l], -1)

            else:
                rgb_h_r = rgb_h[:,:,None]
                input_exps = input_exps[:,:,None].repeat(1,3,1)
                h = rgb_h_r + torch.log(input_exps)
                lnx = torch.reshape(h, [-1, 3])
                h = torch.reshape(h, [-1, 1])
                for i, l in enumerate(self.exps_linears_r):
                    h = self.exps_linears_r[i](h)
                    h = F.relu(h)
                rgb_l = self.rgb_l_linner(h)
                rgb_l = torch.reshape(rgb_l, [-1, 3])

            outputs = torch.cat([rgb_h, rgb_l, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, lnx

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


def draw_CRF(basedir, model):

    ln_x = torch.linspace(-5, 3, 1000).reshape([-1, 1])
    r_h = ln_x
    g_h = ln_x
    b_h = ln_x

    for i, l in enumerate(model.exps_linears_r):
        r_h = model.exps_linears_r[i](r_h)
        r_h = F.relu(r_h)
    r_l = model.r_l_linner(r_h)

    for i, l in enumerate(model.exps_linears_g):
        g_h = model.exps_linears_g[i](g_h)
        g_h = F.relu(g_h)
    g_l = model.g_l_linner(g_h)

    for i, l in enumerate(model.exps_linears_b):
        b_h = model.exps_linears_b[i](b_h)
        b_h = F.relu(b_h)
    b_l = model.b_l_linner(b_h)

    rgb_l = torch.sigmoid(torch.cat([r_l, g_l, b_l], -1))

    x = ln_x.cpu().numpy()
    y = rgb_l.detach().cpu().numpy()

    # simple tone mapper for synthetic dataset
    def tonemapSimple(x):
        return (x / (x + 1)) ** (1 / 2.2)

    z_simple = np.clip(tonemapSimple(np.exp(x)), 0, 1)

    plt.xlabel("lnE + lnt")
    plt.ylabel("pixel value")
    plt.plot(x,y[:,0:1], color='r', label='CRF red')
    plt.plot(x,y[:,1:2], color='g', label='CRF green')
    plt.plot(x,y[:,2:3], color='b', label='CRF blue')
    plt.plot(x, z_simple, color='y', label='CRF GT')
    plt.legend(['CRF red', 'CRF green', 'CRF blue', 'CRF GT'])
    plt.grid()
    plt.savefig(os.path.join(basedir, 'CRF.png'))
    plt.close()

    CRF = np.concatenate([x, y], -1)
    CRF = {'crf_ours':CRF}
    scio.savemat(os.path.join(basedir, 'CRF.mat'), CRF)


def point_constraint(model, gt):
    
    ln_x = torch.zeros([3,1])

    r_h = ln_x
    g_h = ln_x
    b_h = ln_x

    for i, l in enumerate(model.exps_linears_r):
        r_h = model.exps_linears_r[i](r_h)
        r_h = F.relu(r_h)
    r_l = model.r_l_linner(r_h)

    for i, l in enumerate(model.exps_linears_g):
        g_h = model.exps_linears_g[i](g_h)
        g_h = F.relu(g_h)
    g_l = model.g_l_linner(g_h)

    for i, l in enumerate(model.exps_linears_b):
        b_h = model.exps_linears_b[i](b_h)
        b_h = F.relu(b_h)
    b_l = model.b_l_linner(b_h)

    rgb_l = torch.sigmoid(torch.cat([r_l, g_l, b_l], -1))
    
    return img2mse(rgb_l, gt, 1)


def tonemapR(hdr_img):
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    if len(hdr_img.shape) == 4:
        ldr_list = []
        for i in range(hdr_img.shape[0]):
            ldr_list.append(tonemapReinhard.process(hdr_img[i]))
        ldr_img = np.stack(ldr_list, 0)
    else:
        ldr_img = tonemapReinhard.process(hdr_img)

    return ldr_img
