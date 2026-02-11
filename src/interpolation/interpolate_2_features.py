import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import sys
sys.path.append('../nvidia-stylegan2-ada')
from dnnlib import *
from torch_utils import *
sys.path.append('../src')

from interpolate_utils import *

# if you reposition to nvidia/stylegan2-ada
# eg.
# python ffhq-align.py
# python ../nvidia-stylegan2-ada/projector.py --outdir=../data/outputs/img1 --target=../data/processed/align-gen1.jpg --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# ---------------------------------------

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_latent', type=str, default='../data/outputs/img1/projected_w_303.npz' , help='The path to the face image')
    parser.add_argument('--feature_num_1', type=int, default=0)
    parser.add_argument('--feature_num_2', type=int, default=7)
    parser.add_argument('--noise_mode', type=str, default='const', help='Noise mode') #'const', 'random', 'none'
    parser.add_argument('--slider_step', type=float, default=0.5, help='Size of the step for the slider')
    parser.add_argument('--network_pkl', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', help='The URL to the network .pkl file')
    parser.add_argument('--min_val', type=float, default=0.0)
    parser.add_argument('--max_val', type=float, default=10.0)
    args = parser.parse_args()
    return args

def main():
    args = parse()

    w = torch.from_numpy(np.load(args.img_latent)['w']).cuda()

    G = load_network(args.network_pkl)

    all_imgs = calculate_all_images3(
        G, 
        args.slider_step, 
        args.noise_mode, 
        w,
        args.min_val,
        args.max_val,
        args.feature_num_1,
        args.feature_num_2,
    )

    fig, ax = setup_figure(bottom=0.2)

    im = ax.imshow(all_imgs[0, 0])
    feature1_slider = Slider(plt.axes([0.15, 0.05, 0.75, 0.03]), f'feat_{args.feature_num_1} value', args.min_val, args.max_val, valinit=args.min_val, valstep=args.slider_step)
    feature2_slider = Slider(plt.axes([0.15, 0.15, 0.75, 0.03]), f'feat_{args.feature_num_2} value', args.min_val, args.max_val, valinit=args.min_val, valstep=args.slider_step)

    def update(val):
        feature1_value = feature1_slider.val
        feature2_value = feature2_slider.val
        step1 = int((feature1_value - args.min_val) / args.slider_step)
        step2 = int((feature2_value - args.min_val) / args.slider_step)
        new_img = all_imgs[step1, step2]
        im.set_array(new_img)
        fig.canvas.draw_idle()

    feature1_slider.on_changed(update)
    feature2_slider.on_changed(update)
    plt.show()

main()