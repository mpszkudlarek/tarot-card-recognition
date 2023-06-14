import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision
import matplotlib.pyplot as plt
import cv2


data_transform_no_augmentation = transforms.Compose([
    transforms.Resize(size=(224, 348)),
    transforms.ToTensor()
])

data_transform_Trivial = transforms.Compose([
    #Resize the images
    transforms.Resize(size=(224, 348)),
    #Perform random operations
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    #Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

data_transform_Padding = transforms.Compose([
    #add padding
    transforms.Pad(200),
    #Resize the images
    transforms.Resize(size=(224, 348)),
    #Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

data_transform_Perspective = transforms.Compose([
    #Change perspective
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    #Resize the images
    transforms.Resize(size=(224, 348)),
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])

custom_image_tr = tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,348)),
    transforms.ToTensor()
])

data_trnasform_ColorJitter = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5,1.5), contrast=(0.5, 1.5), saturation=(0.5,1.5), hue=0.1),
    transforms.Resize(size=(224, 348)),
    transforms.ToTensor()
])

def extract_class_names_to_file():
    with open(r'names.txt', 'w') as fp:
        for item in class_names:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')