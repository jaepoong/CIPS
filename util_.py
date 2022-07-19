import random
import os
from typing import NamedTuple

import numpy as np
import torch
import torchvision
from torch.utils import data
from PIL import Image

import model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def PCA_plot(data,n_components=3):
    emb_np=g_ema.emb.input.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    pca=PCA(n_components=n_components)
    pca.fit(emb_np)
    df=pca.transform(emb_np)
    im=df.reshape(256,256,3)
    plt.imshow(im)
