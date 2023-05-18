from torch.utils.data import Dataset
import glob
import os
from albumentations.pytorch import ToTensorV2
import torch
import imageio
from PIL import Image
from scipy import ndimage
from torchvision.io import read_image
import cv2
import numpy as np
import albumentations as A
from torchvision.transforms import transforms
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, mask, *args, **kwargs):
        return self.transforms(image=np.array(img), mask = np.array(mask))

class BasicDataset(Dataset):
    def __init__(self, images_dir,masks_dir, transform=None):
        self.images = sorted(glob.glob(images_dir+ "/*jpg"))
        self.masks = sorted(glob.glob(masks_dir+ "/*png"))
        self.transform = Transforms(transform)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, indx):
        # Apply transformations if specified
        image = cv2.imread(self.images[indx], cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[indx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            example = self.transform(image,mask)
            image = example['image'].float()
            mask  = example['mask']
       # Convert PIL image to PyTorch tensor
        #image = transforms.ToTensor()(image)
        #mask = transforms.ToTensor()(mask)
        # Normalize image tensor
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        threshold_value = 128
        mask = torch.where(mask > threshold_value, torch.tensor(1), torch.tensor(0))
        image = normalize(image)
        return{
                'image': image,
                'mask': mask
        }

