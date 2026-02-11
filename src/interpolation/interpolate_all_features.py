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
    parser.add_argument('--feature_nums', nargs='+', type=int, default=[0, 7, 5])
    parser.add_argument('--noise_mode', type=str, default='const', help='Noise mode') #'const', 'random', 'none'
    parser.add_argument('--slider_step', type=float, default=0.25, help='Size of the step for the slider')
    parser.add_argument('--network_pkl', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', help='The URL to the network .pkl file')
    parser.add_argument('--min_val', type=float, default=-1.0)
    parser.add_argument('--max_val', type=float, default=4.0)
    args = parser.parse_args()
    return args

def main():
    args = parse()

    w = torch.from_numpy(np.load(args.img_latent)['w']).cuda()

    G = load_network(args.network_pkl)

    fig, ax = setup_figure(bottom=0.1 + len(args.feature_nums) * 0.1)

    im = ax.imshow(calculate_image(G, w, args.noise_mode))
    feature_sliders = []
    for i, f_num in enumerate(args.feature_nums):
        f_slider = Slider(plt.axes([0.15, 0.05 + i*0.1, 0.75, 0.03]), f'feat_{f_num} value', args.min_val, args.max_val, valinit=args.min_val, valstep=args.slider_step)
        feature_sliders.append(f_slider)

    def update(val):
        w_copy = w[:, :, :]
        for i, f_num in enumerate(args.feature_nums):
            w_copy[0, f_num] = feature_sliders[i].val
        new_img = calculate_image(G, w_copy, args.noise_mode)
        im.set_array(new_img)
        fig.canvas.draw_idle()

    for f in feature_sliders:
        f.on_changed(update)
    
    update(None)
    plt.show()

main()