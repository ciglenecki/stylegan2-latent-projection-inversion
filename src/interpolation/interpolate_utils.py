import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dnnlib import *
from legacy import load_network_pkl
from torch_utils import *
from tqdm import tqdm
import matplotlib as mpl

def load_network(url):
    print(f'Loading network from {url}...')
    with util.open_url(url) as fp:
        G = load_network_pkl(fp)['G'].requires_grad_(False).cuda()
    print('Network loaded...')
    return G

def calculate_image(G, w, noise_mode):
    img = G.synthesis(w, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_numpy = img.cpu().numpy().reshape(1024, 1024, 3)
    return img_numpy

def calculate_all_images(G, slider_step, noise_mode, w1, w2):
    num_steps = int(1 / slider_step) + 1
    all_imgs = []
    for i in tqdm(range(num_steps)):
        k1, k2 = i * slider_step, 1 - i * slider_step 
        img_i = calculate_image(G, w1 * k1 + w2 * k2, noise_mode)
        all_imgs.append(img_i)
    return np.array(all_imgs)

def calculate_all_images2(G, slider_step, noise_mode, w, min_val, max_val, feature_num=0):
    num_steps = int((max_val - min_val) / slider_step) + 1
    all_imgs = []

    curr_min = min_val
    for i in tqdm(range(num_steps)):
        w_i = w.clone()
        w_i[0, feature_num] = w_i[0, feature_num] + curr_min
        all_imgs.append(calculate_image(G, w_i, noise_mode))
        curr_min += slider_step
    return np.array(all_imgs)

def calculate_all_images3(G, slider_step, noise_mode, w, min_val, max_val, feature_num_1, feature_num_2):
    num_steps = int((max_val - min_val) / slider_step) + 1
    all_imgs = []

    curr_min_outer = min_val
    for _ in tqdm(range(num_steps)):
        curr_min_inner = min_val
        tmp = []
        for _ in range(num_steps):
            w_i = w.clone()
            w_i[0, feature_num_1] = w_i[0, feature_num_1] + curr_min_outer
            w_i[0, feature_num_2] = w_i[0, feature_num_2] + curr_min_inner
            tmp.append(calculate_image(G, w_i, noise_mode))
            curr_min_inner += slider_step
        curr_min_outer += slider_step
        all_imgs.append(tmp)
    return np.array(all_imgs)

def setup_figure(bottom=0.1):
    mpl.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Interpolation demo')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(bottom=bottom)
    return fig, ax