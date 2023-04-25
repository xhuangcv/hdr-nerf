# HDR-NeRF: High Dynamic Range Neural Radiance Fields
### [Project Page](https://xhuangcv.github.io/hdr-nerf/) | [Video](https://youtu.be/GmxsW9L1O6s) | [Paper](https://arxiv.org/abs/2111.14451) | [Dataset](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I?usp=sharing)

We present High Dynamic Range Neural Radiance Fields (HDR-NeRF) to recover an HDR radiance field from a set of low dynamic range (LDR) views with different exposures. Using the HDR-NeRF, we are able to generate both novel HDR views and novel LDR views under different exposures.

![](https://github.com/shsf0817/hdr-nerf/blob/gh-pages/images/overview.png)

## Method Overview
The pipeline of HDR-NeRF modeling the simplified physical process. Our method is consisted of two modules: an HDR radiance field models the scene for radiance and densities and a tone mapper models the CRF for colors.

<p align="center">
<img src="https://github.com/shsf0817/hdr-nerf/blob/gh-pages/images/pipeline.png" style="width:512px;"/>
</p>

## Quick Start
### Setup
```
git clone https://github.com/shsf0817/hdr-nerf.git
cd hdr-nerf
pip install -r requirements.txt
```
<details>
  <summary> Dependencies (click to expand) </summary>
  
   - torch==1.9.0
   - torchvision==0.10.0
   - numpy==1.19.0
   - imageio-ffmpeg==0.4.5
   - imageio==2.9.0
   - opencv-python==4.5.3.56
   - tqdm==4.62.2
   - scikit-image==0.17.2
   - ConfigArgParse==1.5.2
  
</details>
We provide a pip environment setup file including all of the above dependencies. To read and write EXR files, run following commands at Python terminal:

```
    import imageio
    imageio.plugins.freeimage.download()
```

### Download dataset
We collect an HDR dataset (multi-view and multi-exposure) that contains 8 synthetic scenes rendered with [Blender](https://www.blender.org/) and 4 real scenes captured by a digital camera. Images are collected at 35 different poses in the real dataset, with 5 different exposure time $\{t_1, t_2, t_3, t_4, t_5\}$ at each pose. You can download all our dataset in [here](https://drive.google.com/drive/folders/1OTDLLH8ydKX1DcaNpbQ46LlP0dKx6E-I?usp=sharing). 

For a quick demo, please download ```demo``` folder and move it to ```hdr-nerf``` folder.

### Render a demo
```
python3 run_nerf.py --config configs/demo.txt --render_only
```
Both LDR and HDR results are saved in  ```<basedir>/<expname>_<render_out_path>``` . All HDR results in the experiment are tonemapped using [Phototmatix](https://www.hdrsoft.com/). Please install [Phototmatix](https://www.hdrsoft.com/) or [Luminance HDR](http://qtpfsgui.sourceforge.net/) for the visualization of HDR results.
## Train HDR-NeRF
```
python3 run_nerf.py --config configs/flower.txt
```
Intermediate results and models are saved in ```<basedir>/<expname>```

## Cite

```
@article{huang2021hdr,
  title={HDR-NeRF: High Dynamic Range Neural Radiance Fields},
  author={Huang, Xin and Zhang, Qi and Ying, Feng and Li, Hongdong and Wang, Xuan and Wang, Qing},
  journal={arXiv preprint arXiv:2111.14451},
  year={2021}
}
```

## Acknowledge
Our code is based on the famous pytorch reimplementation of NeRF, [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch/). We appreciate all the contributors.
