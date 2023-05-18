from torch.utils.data import Dataset
import glob
import os
import torch
import imageio
from PIL import Image
from scipy import ndimage
from torchvision.io import read_image
from torchvision.transforms import transforms
class BasicDataset(Dataset):
    def __init__(self, images_dir,masks_dir, transform=None):
        self.images = sorted(glob.glob(images_dir+ "/*jpg"))
        self.masks = sorted(glob.glob(masks_dir+ "/*png"))
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, indx):
        # Apply transformations if specified
        image = image = Image.open(self.images[indx])
        mask = Image.open(self.masks[indx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
       # Convert PIL image to PyTorch tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        # Normalize image tensor
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        return{
                'image': image,
                'mask': mask
        }

