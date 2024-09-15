import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def predict_first(image_path, model, img_tensor, topk=5, category_names=None, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    #process image
    img_tensor = img_tensor.unsqueeze(0)

    if gpu:
        model.cuda()
        img_tensor = img_tensor.cuda()

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)

    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)

    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[c.item()] for c in top_class[0]]

    # If category names are provided, map classes to names
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name.get(str(cls), cls) for cls in top_classes]

    return top_p[0].tolist(), top_classes
